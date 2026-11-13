import copy, json, math, unittest
import numpy as np
from src.services.seed import demo_rows, POOLS, STOP
from src.services.statistics import analyze
from src.services.projection import project
from src.services.tokenize import tokenize

LEXICON={'id':'lex-test','name':'测试词典','version':'test-v1','terms':[t for p in POOLS for t in p]+STOP,'stopwords':STOP}
STUDY={'topic_count':6,'seed':2305,'max_iter':250,'tolerance':1e-5}


def make_run(rows=None):
    rows=demo_rows() if rows is None else rows
    return {'id':'run-test','study_id':'study-test','status':'completed','error':'',
            'result':analyze(rows,LEXICON,STUDY),'created_at':'2026-11-13T00:00:00+00:00'}


class ProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo_run=make_run()
        cls.docs=[{'doc_id':'SUP-1','text':'我们提到公交、换乘和步行。这个公交需要候车，然后我们感觉换乘值得讨论。'},
                  {'doc_id':'SUP-2','text':'这里没有任何受控词项。'},
                  {'doc_id':'SUP-3','text':'照护、陪伴与长者，这个照护需要休息。'},
                  {'doc_id':'SUP-4','text':'公交 公交 公交'}]
        cls.result=project(cls.demo_run,documents=cls.docs)
        cls.by_id={d['doc_id']:d for d in cls.result['documents']}

    def test_basis_is_taken_verbatim_from_completed_run(self):
        basis=self.result['basis'];result=self.demo_run['result']
        self.assertEqual(basis['topic_ids'],['T01','T02','T03','T04','T05','T06'])
        self.assertEqual(basis['vocabulary_size'],len(result['vocabulary']))
        self.assertEqual(basis['lexicon']['version'],'test-v1')
        self.assertEqual(basis['original_document_count'],result['included'])
        self.assertEqual(basis['idf_source'],'原运行持久化idf，未重算')

    def test_weights_are_non_negative_and_finite_with_expected_shape(self):
        for doc in self.result['documents']:
            weights=doc['weights']
            self.assertEqual(len(weights),self.demo_run['result']['nmf']['topic_count'])
            self.assertTrue(all(math.isfinite(w) and w>=0 for w in weights))
            self.assertAlmostEqual(sum(doc['display_fractions']),1.0,places=9)

    def test_reconstruction_error_matches_reported_weights(self):
        h=np.array(self.demo_run['result']['nmf']['h'])
        idf=self.demo_run['result']['idf'];vocabulary=self.demo_run['result']['vocabulary']
        for doc in self.result['documents']:
            tokens=tokenize(dict((d['doc_id'],d['text']) for d in self.docs)[doc['doc_id']],LEXICON['terms'],LEXICON['stopwords'])
            vector=np.zeros(len(vocabulary))
            for i,term in enumerate(vocabulary): vector[i]=tokens['counts'].get(term,0)/len(tokens['tokens'])*idf[i]
            expected=float(np.linalg.norm(vector-np.array(doc['weights'])@h))
            self.assertAlmostEqual(doc['reconstruction_error'],expected,places=9)
            self.assertAlmostEqual(doc['relative_reconstruction_error'],expected/np.linalg.norm(vector),places=9)

    def test_neighbors_come_from_original_valid_documents_only(self):
        original_ids={d['doc_id'] for d in self.demo_run['result']['documents']}
        for doc in self.result['documents']:
            self.assertEqual(len(doc['neighbors']),3)
            for neighbor in doc['neighbors']:
                self.assertIn(neighbor['doc_id'],original_ids)
                self.assertGreaterEqual(neighbor['similarity'],0)
                self.assertLessEqual(neighbor['similarity'],1)
            keys=[(-round(n['similarity'],12),n['doc_id']) for n in doc['neighbors']]
            self.assertEqual(keys,sorted(keys))

    def test_no_valid_terms_is_reported_without_weights(self):
        reasons={d['doc_id']:d['reason'] for d in self.result['isolated']}
        self.assertIn('SUP-2',reasons)
        self.assertIn('无有效受控词项',reasons['SUP-2'])
        self.assertNotIn('SUP-2',self.by_id)
        self.assertEqual(self.result['summary']['included'],3)
        self.assertEqual(self.result['summary']['isolated'],1)

    def test_split_batches_and_order_do_not_change_weights(self):
        docs=[self.docs[3],self.docs[0],self.docs[2]]
        reversed_batch=project(self.demo_run,documents=list(reversed(docs)))
        single_2=project(self.demo_run,documents=[self.docs[2]])
        single_4=project(self.demo_run,documents=[self.docs[3]])
        first=project(self.demo_run,documents=self.docs)
        lookups=[(reversed_batch,0),(reversed_batch,1),(reversed_batch,2)]
        self.assertEqual(reversed_batch['documents'][0]['weights'],single_2['documents'][0]['weights'])
        self.assertEqual(reversed_batch['documents'][2]['weights'],single_4['documents'][0]['weights'])
        for batch,index in lookups:
            doc=batch['documents'][index]
            self.assertEqual(doc['weights'],self.by_id[doc['doc_id']]['weights'])
            self.assertEqual(doc['neighbors'],self.by_id[doc['doc_id']]['neighbors'])
            self.assertEqual(doc['reconstruction_error'],self.by_id[doc['doc_id']]['reconstruction_error'])
        self.assertEqual(first['documents'][0]['weights'],project(self.demo_run,documents=[self.docs[0]])['documents'][0]['weights'])

    def test_csv_and_json_inputs_agree(self):
        csv_text='doc_id,text\nSUP-9,"我们提到公交、换乘和步行。这个公交需要候车，然后我们感觉换乘值得讨论。"\n'
        from_csv=project(self.demo_run,csv_text=csv_text)
        from_json=project(self.demo_run,documents=[{'doc_id':'SUP-9','text':'我们提到公交、换乘和步行。这个公交需要候车，然后我们感觉换乘值得讨论。'}])
        self.assertEqual(from_csv['documents'][0]['weights'],from_json['documents'][0]['weights'])
        self.assertEqual(from_csv['documents'][0]['neighbors'],from_json['documents'][0]['neighbors'])

    def test_unknown_terms_count_in_tf_denominator_but_not_features(self):
        doc=self.by_id['SUP-1']
        self.assertGreater(doc['unknown_characters'],0)
        self.assertEqual(doc['token_count'],len(tokenize(self.docs[0]['text'],LEXICON['terms'],LEXICON['stopwords'])['tokens']))

    def test_runs_without_complete_basis_are_explicitly_rejected(self):
        def broken(mutator):
            run=copy.deepcopy(self.demo_run); mutator(run); return run
        cases={
            'not completed':lambda r:r.update(status='failed'),
            'no idf':lambda r:r['result'].pop('idf'),
            'no df':lambda r:r['result'].pop('df'),
            'no h':lambda r:r['result']['nmf'].pop('h'),
            'no sparse':lambda r:r['result'].pop('sparse_tfidf'),
            'no lexicon snapshot':lambda r:r['result'].pop('lexicon_snapshot'),
            'no result':lambda r:r.update(result={}),
            'idf tampered':lambda r:r['result']['idf'].__setitem__(0,r['result']['idf'][0]*1.01),
            'h negative':lambda r:r['result']['nmf']['h'].__setitem__(0,-1),
            'h zero row':lambda r:r['result']['nmf']['h'].__setitem__(0,[0.0]*len(r['result']['vocabulary'])),
            'h column mismatch':lambda r:r['result']['nmf']['h'].__setitem__(0,r['result']['nmf']['h'][0][:-1]),
            'vocabulary mismatch':lambda r:r['result']['sparse_tfidf'][0]['nonzero'].__setitem__(0,(400,1.0)),
            'duplicate vocabulary':lambda r:r['result']['vocabulary'].append(r['result']['vocabulary'][0]),
            'duplicate topic ids':lambda r:r['result']['topics'].__setitem__(1,{'id':'T01'}),
        }
        for label,mutator in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    project(broken(mutator),documents=[{'doc_id':'SUP-1','text':'公交'}])
                self.assertIn('计算依据',str(caught.exception))

    def test_document_level_input_errors_are_rejected(self):
        bad=[
            [{'doc_id':'','text':'公交'}],
            [{'doc_id':'SUP-1','text':''}],
            [{'doc_id':'SUP-1','text':'公交'},{'doc_id':'SUP-1','text':'换乘'}],
            [{'doc_id':'LW-0001','text':'公交'}],
            [{'doc_id':'SUP-1','text':'公交'*1001}],
        ]
        for documents in bad:
            with self.subTest(documents=documents):
                with self.assertRaises(ValueError):
                    project(self.demo_run,documents=documents)
        with self.assertRaises(ValueError):
            project(self.demo_run,documents=[])
        with self.assertRaises(ValueError):
            project(self.demo_run,documents=[{'doc_id':'SUP-1','text':'公交'}],csv_text='doc_id,text\nSUP-1,公交\n')

    def test_original_run_and_review_are_not_modified(self):
        run=make_run(); before=json.dumps(run,sort_keys=True,ensure_ascii=False)
        project(run,documents=[{'doc_id':'SUP-1','text':'公交换乘'}])
        self.assertEqual(before,json.dumps(run,sort_keys=True,ensure_ascii=False))


if __name__=='__main__':
    unittest.main(verbosity=2)


class LegacyRunCompatibilityTest(unittest.TestCase):
    """旧运行JSON只要数值计算依据齐全就兼容；缺任何一项则明确拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.legacy_run={'id':'run-legacy','study_id':'study-test','status':'completed','error':'','result':analyze(demo_rows(),LEXICON,STUDY)}

    def test_legacy_payload_without_presentation_fields_still_projects_identically(self):
        legacy=copy.deepcopy(self.legacy_run)
        for key in ('checksum','limitations','unused_terms','topics','documents','isolated'):
            legacy['result'].pop(key,None)
        legacy['result'].pop('included',None)
        document=[{'doc_id':'OLD-1','text':'我们提到公交、换乘和步行。'}]
        current=project(self.legacy_run,documents=document)
        projected=project(legacy,documents=document)
        self.assertEqual(projected['documents'][0]['weights'],current['documents'][0]['weights'])
        self.assertEqual([t['id'] for t in [{'id':x} for x in projected['basis']['topic_ids']]],['T01','T02','T03','T04','T05','T06'])

    def test_legacy_payload_missing_numeric_basis_is_rejected_not_backfilled(self):
        for key in ('idf','df','vocabulary','sparse_tfidf','lexicon_snapshot'):
            with self.subTest(key=key):
                legacy=copy.deepcopy(self.legacy_run)
                legacy['result'].pop(key)
                with self.assertRaises(ValueError) as caught:
                    project(legacy,documents=[{'doc_id':'OLD-1','text':'公交'}])
                self.assertIn('计算依据',str(caught.exception))


class BoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lexicon={**LEXICON,'terms':LEXICON['terms']+['议价']}
        rows=demo_rows()
        cls.boundary_run={'id':'run-boundary','status':'completed','result':analyze(rows,lexicon,STUDY)}
        cls.base='我们提到公交、换乘和步行。'

    def test_document_count_boundaries(self):
        docs=[{'doc_id':f'B-{i}','text':self.base} for i in range(600)]
        result=project(self.boundary_run,documents=docs)
        self.assertEqual(result['summary']['included'],600)
        with self.assertRaises(ValueError): project(self.boundary_run,documents=docs+[{'doc_id':'B-600','text':self.base}])

    def test_text_and_id_length_boundaries(self):
        text=('公交'*1000)[:2000]
        self.assertEqual(len(text),2000)
        result=project(self.boundary_run,documents=[{'doc_id':'B-'+'x'*198,'text':text}])
        self.assertEqual(result['summary']['included'],1)
        with self.assertRaises(ValueError): project(self.boundary_run,documents=[{'doc_id':'B-x','text':text+'多'}])
        with self.assertRaises(ValueError): project(self.boundary_run,documents=[{'doc_id':'B-'+'x'*199,'text':self.base}])
        with self.assertRaises(ValueError): project(self.boundary_run,documents=[{'doc_id':'B-x','text':' '}])

    def test_controlled_terms_outside_original_vocabulary(self):
        result=project(self.boundary_run,documents=[{'doc_id':'OUT-1','text':'议价'}])
        self.assertEqual(result['summary']['included'],0)
        self.assertEqual(result['isolated'][0]['out_of_vocabulary_terms'],['议价'])
        mixed=project(self.boundary_run,documents=[{'doc_id':'OUT-2','text':'公交议价'}])
        document=mixed['documents'][0]
        self.assertEqual(document['out_of_vocabulary_terms'],['议价'])
        # 词表外受控词计入TF分母：公交只占一半token，权重相应减半
        reference=project(self.boundary_run,documents=[{'doc_id':'OUT-3','text':'公交'}])['documents'][0]
        self.assertEqual(document['token_count'],2)
        self.assertEqual([round(a,12) for a in document['weights']],[round(b/2,12) for b in reference['weights']])
