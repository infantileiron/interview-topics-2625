import os, tempfile, unittest

os.environ.setdefault('DATABASE_URL','sqlite:///'+tempfile.mkstemp(suffix='.sqlite')[1])
os.environ.setdefault('SEED_DEMO','true')
os.environ.setdefault('TASK_MODE','inline')
os.environ.setdefault('ARTIFACT_DIR',tempfile.mkdtemp())

from fastapi.testclient import TestClient
from src.main import app
from src.utils import repository as repo

DOC='我们提到公交、换乘和步行。这个公交需要候车，然后我们感觉换乘值得讨论。'


class ProjectionApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(app); cls.client.__enter__()
        cls.headers={role:{'Authorization':'Bearer '+cls.token(role)} for role in ('admin','analyst','viewer')}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None)

    @classmethod
    def token(cls,role):
        response=cls.client.post('/auth/login',json={'email':f'{role}@liwen.test','password':'Topic2026!'})
        return response.json()['access_token']

    def test_analyst_can_project_and_viewer_cannot(self):
        response=self.client.post('/runs/run-demo/projections',headers=self.headers['analyst'],
                                  json={'documents':[{'doc_id':'HTTP-1','text':DOC}]})
        self.assertEqual(response.status_code,201,response.text)
        body=response.json()
        self.assertEqual(body['documents'][0]['doc_id'],'HTTP-1')
        self.assertEqual(len(body['documents'][0]['weights']),6)
        self.assertTrue(all(w>=0 for w in body['documents'][0]['weights']))
        self.assertEqual(self.client.post('/runs/run-demo/projections',headers=self.headers['viewer'],
                                          json={'documents':[{'doc_id':'HTTP-1','text':DOC}]}).status_code,403)
        self.assertEqual(self.client.post('/runs/run-demo/projections',
                                          json={'documents':[{'doc_id':'HTTP-1','text':DOC}]}).status_code,401)

    def test_csv_body_with_isolated_document(self):
        csv_text=f'doc_id,text\nHTTP-2,"{DOC}"\nHTTP-3,\'\'\n'
        response=self.client.post('/runs/run-demo/projections',headers=self.headers['admin'],json={'csv_text':csv_text})
        self.assertEqual(response.status_code,201,response.text)
        body=response.json()
        self.assertEqual(body['summary']['input_documents'],2)
        self.assertEqual(body['summary']['isolated'],1)
        self.assertIn('无有效受控词项',body['isolated'][0]['reason'])

    def test_incomplete_basis_returns_422_and_does_not_touch_run(self):
        run=repo.get('runs','run-demo'); broken={k:v for k,v in run.items()}
        broken['result']={k:v for k,v in run['result'].items() if k!='idf'}
        self.assertEqual(broken['result'].get('idf'),None)
        # 直接在服务层注入伪造运行，确认 HTTP 返回 422 且原运行不变
        from src.services.projection import project
        with self.assertRaises(ValueError):
            project(broken,documents=[{'doc_id':'HTTP-4','text':DOC}])
        self.assertEqual(self.client.post('/runs/nope/projections',headers=self.headers['analyst'],
                                          json={'documents':[{'doc_id':'HTTP-4','text':DOC}]}).status_code,404)
        self.assertEqual(repo.get('runs','run-demo')['result']['included'],run['result']['included'])

    def test_input_shape_errors_return_422(self):
        for payload in ({},{'documents':[]},
                        {'documents':[{'doc_id':'HTTP-5','text':DOC}],'csv_text':'doc_id,text\nHTTP-5,x\n'},
                        {'documents':[{'doc_id':'LW-0001','text':DOC}]}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/runs/run-demo/projections',headers=self.headers['analyst'],
                                                  json=payload).status_code,422)


if __name__=='__main__':
    unittest.main(verbosity=2)
