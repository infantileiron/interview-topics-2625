import csv, hashlib, io, json, math
import numpy as np
from src.constants.domain import EPSILON, MAX_CHARACTERS, MAX_DOCUMENTS
from src.services.tokenize import tokenize

# 固定基底的补访投影：不更新H、不重算IDF/词项顺序、不写入原运行或人工命名。
PROJECTION_MAX_ITER=2000
PROJECTION_MIN_ITER=10
PROJECTION_TOLERANCE=1e-10
NEIGHBOR_LIMIT=3
MAX_DOC_ID=200


def _fail(problems):
    raise ValueError('该运行不具备补访投影的计算依据：'+'；'.join(problems)+'。不会重新训练、补算或改写原运行，请改用计算依据齐全的已完成运行。')


def _is_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(float(value))


def _stable(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def build_basis(run):
    """校验并返回原运行中已持久化的固定计算依据；缺失或不一致时明确拒绝。"""
    problems=[]
    if not isinstance(run,dict): _fail(['运行记录不是对象'])
    if run.get('status')!='completed': problems.append(f"运行状态为{run.get('status') or '未知'}，仅completed运行可作为基线")
    result=run.get('result')
    if not isinstance(result,dict) or not result: problems.append('缺少已完成的运行结果JSON')
    if problems: _fail(problems)

    lexicon=result.get('lexicon_snapshot')
    if not isinstance(lexicon,dict):
        problems.append('缺少原运行词典快照lexicon_snapshot')
    else:
        terms=lexicon.get('terms'); stopwords=lexicon.get('stopwords')
        if not isinstance(terms,list) or not terms: problems.append('词典快照缺少terms')
        elif any(not isinstance(t,str) or not t for t in terms): problems.append('词典快照terms含空值或非字符串')
        elif len(set(terms))!=len(terms): problems.append('词典快照terms存在重复项')
        if not isinstance(stopwords,list): problems.append('词典快照缺少stopwords')
        elif any(not isinstance(t,str) or not t for t in stopwords): problems.append('词典快照stopwords含空值或非字符串')
        elif len(set(stopwords))!=len(stopwords): problems.append('词典快照stopwords存在重复项')
        elif isinstance(terms,list) and not set(stopwords).issubset(set(terms)): problems.append('词典快照停用词不在词表内')
        elif isinstance(terms,list) and not set(terms)-set(stopwords): problems.append('词典快照移除停用词后为空')

    vocabulary=result.get('vocabulary')
    if not isinstance(vocabulary,list) or not vocabulary:
        problems.append('缺少原运行特征词表vocabulary')
        vocabulary=[]
    else:
        if any(not isinstance(t,str) or not t.strip() for t in vocabulary): problems.append('原运行vocabulary含空值或非字符串')
        if len(set(vocabulary))!=len(vocabulary): problems.append('原运行vocabulary存在重复词项')

    idf=result.get('idf')
    if not isinstance(idf,list) or not idf:
        problems.append('缺少原运行逆文档频率idf')
        idf=[]
    elif any(not _is_number(v) or float(v)<=0 for v in idf):
        problems.append('原运行idf含非有限、非正或非数值项')
    elif len(vocabulary) and len(idf)!=len(vocabulary): problems.append('原运行idf长度与vocabulary不一致')
    if idf and not any(not _is_number(v) or float(v)<=0 for v in idf):
        df=result.get('df')
        if not isinstance(df,list) or len(df)!=len(idf):
            problems.append('缺少与原运行idf对应的文档频率df，无法核对计算依据')
        elif any(not isinstance(v,int) or isinstance(v,bool) or not 0<=v<=len(result.get('sparse_tfidf') or []) for v in df):
            problems.append('原运行df含无效或越界的文档频率')
        elif not np.allclose([math.log((1+len(result.get('sparse_tfidf') or []))/(1+v))+1 for v in df],np.asarray(idf,dtype=float),rtol=1e-9,atol=1e-12):
            problems.append('原运行idf与持久化df/有效文档数不一致，无法确认计算依据')
    if vocabulary and isinstance(lexicon,dict) and isinstance(lexicon.get('terms'),list) and isinstance(lexicon.get('stopwords'),list):
        effective=set(lexicon['terms'])-set(lexicon['stopwords'])
        outside=[term for term in vocabulary if term not in effective]
        if outside: problems.append('原运行vocabulary含词典有效词表之外的词项：'+'、'.join(outside[:5]))

    nmf=result.get('nmf')
    h_raw=nmf.get('h') if isinstance(nmf,dict) else None
    h=None
    if not isinstance(h_raw,list) or not h_raw:
        problems.append('缺少原运行主题基底nmf.h')
    else:
        try: candidate=np.asarray(h_raw,dtype=float)
        except (TypeError,ValueError): candidate=None
        if candidate is None or candidate.ndim!=2 or not candidate.size:
            problems.append('原运行nmf.h不是非空二维数值矩阵')
        elif not np.isfinite(candidate).all(): problems.append('原运行nmf.h含非有限值')
        elif np.any(candidate<0): problems.append('原运行nmf.h含负值')
        elif np.any(np.linalg.norm(candidate,axis=1)<=0): problems.append('原运行nmf.h存在全零主题行，主题权重不可识别')
        elif len(vocabulary) and candidate.shape[1]!=len(vocabulary): problems.append('原运行nmf.h列数与vocabulary不一致')
        else: h=candidate
        if h is not None and isinstance(nmf,dict) and 'topic_count' in nmf:
            count=nmf.get('topic_count')
            if not isinstance(count,int) or isinstance(count,bool) or count!=h.shape[0]: problems.append('原运行nmf.topic_count与H行数不一致')

    topic_ids=[]
    topics=result.get('topics')
    if topics is not None:
        if not isinstance(topics,list): problems.append('原运行topics不是数组')
        elif h is not None and len(topics)!=h.shape[0]: problems.append('原运行topics数量与H行数不一致')
        else:
            for topic in topics:
                topic_id=topic.get('id') if isinstance(topic,dict) else None
                if not isinstance(topic_id,str) or not topic_id: problems.append('原运行topics含无效主题编号'); break
                topic_ids.append(topic_id)
            if len(set(topic_ids))!=len(topic_ids): problems.append('原运行topics主题编号重复')
    if h is not None and not topic_ids: topic_ids=[f'T{k+1:02d}' for k in range(h.shape[0])]

    sparse=result.get('sparse_tfidf')
    original_ids=[]; original=None
    if not isinstance(sparse,list) or not sparse:
        problems.append('缺少原语料稀疏TF-IDF向量sparse_tfidf')
    elif idf and len(idf):
        original=np.zeros((len(sparse),len(idf)),dtype=float)
        for row_index,row in enumerate(sparse):
            if not isinstance(row,dict): problems.append(f'原语料第{row_index+1}条稀疏向量不是对象'); break
            doc_id=row.get('doc_id')
            if not isinstance(doc_id,str) or not doc_id.strip() or len(doc_id.strip())>MAX_DOC_ID:
                problems.append(f'原语料第{row_index+1}条doc_id无效'); break
            if doc_id.strip() in original_ids: problems.append(f'原语料doc_id重复：{doc_id.strip()}'); break
            original_ids.append(doc_id.strip())
            pairs=row.get('nonzero')
            if not isinstance(pairs,list) or not pairs: problems.append(f'原语料{doc_id.strip()}缺少非零特征'); break
            seen=set()
            for pair in pairs:
                if not isinstance(pair,(list,tuple)) or len(pair)!=2 or isinstance(pair[0],bool) or not isinstance(pair[0],int):
                    problems.append(f'原语料{doc_id.strip()}的稀疏特征索引无效'); break
                j,value=pair
                if not 0<=j<len(idf) or j in seen or not _is_number(value) or float(value)<=0:
                    problems.append(f'原语料{doc_id.strip()}的稀疏特征越界、重复或权重无效'); break
                seen.add(j); original[row_index,j]=float(value)
            else: continue
            break
        if not problems and original is not None and np.any(np.linalg.norm(original,axis=1)<=0):
            problems.append('原语料稀疏向量存在全零行，不能作为近邻依据')
        if not problems and original is not None:
            df=result.get('df'); recomputed=None
            if isinstance(df,list) and len(df)==len(vocabulary) and all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in df):
                n=len(original_ids); recomputed=np.array([math.log((1+n)/(1+v))+1 for v in df],dtype=float)
            if recomputed is not None and not np.allclose(recomputed,np.asarray(idf,dtype=float),rtol=1e-9,atol=1e-12):
                problems.append('原运行idf与持久化df/有效文档数不一致，无法确认计算依据')
            else:
                # 每个非零TF–IDF值必须能由正词频/有效token数与对应idf复算出来（相对容差宽松以兼容浮点文本化）。
                bad=None
                for row_index,row in enumerate(sparse):
                    values=dict((int(pair[0]),float(pair[1])) for pair in row['nonzero'])
                    token_count=row.get('token_count')
                    if not isinstance(token_count,int) or isinstance(token_count,bool) or token_count<=0:
                        bad=f'原语料{row.get("doc_id")}缺少有效token_count'; break
                    for j,value in values.items():
                        ratio=value/float(idf[j])
                        if ratio*token_count<=0 or abs(round(ratio*token_count)-ratio*token_count)>1e-6:
                            bad=f'原语料{row.get("doc_id")}的TF–IDF权重不能由词频/有效token总数与idf复算'; break
                    if bad: break
                if bad: problems.append(bad)
    if problems: _fail(problems)

    if result.get('documents') is not None:
        documents=result.get('documents')
        if not isinstance(documents,list):
            _fail(['原运行documents不是数组'])
        document_ids=[]
        for item in documents:
            doc_id=item.get('doc_id') if isinstance(item,dict) else None
            if not isinstance(doc_id,str) or not doc_id.strip(): _fail(['原运行documents含无效doc_id'])
            document_ids.append(doc_id.strip())
        if len(set(document_ids))!=len(document_ids): _fail(['原运行documents的doc_id重复'])
        if set(document_ids)!=set(original_ids): _fail(['原运行documents与sparse_tfidf的doc_id不一致'])
    included=result.get('included')
    if included is not None and (not isinstance(included,int) or isinstance(included,bool) or included!=len(original_ids)):
        _fail(['原运行included与sparse_tfidf行数不一致'])

    isolated_ids=set()
    isolated=result.get('isolated')
    if isinstance(isolated,list):
        for item in isolated:
            doc_id=item.get('doc_id') if isinstance(item,dict) else None
            if not isinstance(doc_id,str) or not doc_id.strip(): _fail(['原运行isolated含无效doc_id'])
            doc_id=doc_id.strip()
            if doc_id in isolated_ids or doc_id in original_ids: _fail([f'原运行isolated的doc_id与有效文档或自身重复：{doc_id}'])
            isolated_ids.add(doc_id)
    elif isolated is not None: _fail(['原运行isolated不是数组'])

    basis={'lexicon':lexicon,'vocabulary':vocabulary,'idf':np.asarray(idf,dtype=float),'h':h,'topic_ids':topic_ids,
           'original_ids':original_ids,'original_matrix':original,'original_norms':np.linalg.norm(original,axis=1),
           'original_all_ids':set(original_ids)|isolated_ids}
    basis['checksum']=hashlib.sha256(_stable({'vocabulary':vocabulary,'idf':[float(x) for x in idf],
        'h':h.tolist(),'lexicon':lexicon,'original_ids':original_ids}).encode()).hexdigest()
    return basis


def parse_csv(csv_text):
    if not isinstance(csv_text,str) or not csv_text.strip(): raise ValueError('补访CSV不能为空')
    reader=csv.DictReader(io.StringIO(csv_text.lstrip('﻿')))
    fields=set(reader.fieldnames or [])
    if not {'doc_id','text'}.issubset(fields): raise ValueError('补访CSV需要doc_id,text两列，内容须为匿名补访文本')
    documents=[]
    for line_number,row in enumerate(reader,2):
        if line_number>MAX_DOCUMENTS+1: raise ValueError(f'单次补访投影最多{MAX_DOCUMENTS}篇文档')
        doc_id=(row.get('doc_id') or '').strip()
        text=(row.get('text') or '').strip()
        if not doc_id or not 1<=len(text)<=MAX_CHARACTERS:
            raise ValueError(f'补访CSV第{line_number}行doc_id为空或文本不在1—{MAX_CHARACTERS}字之间')
        documents.append({'doc_id':doc_id,'text':text})
    if not documents: raise ValueError('补访CSV至少需要1篇有效文档')
    return documents


def validate_documents(documents,original_ids):
    if not isinstance(documents,list) or not 1<=len(documents)<=MAX_DOCUMENTS:
        raise ValueError(f'单次补访投影须提供1—{MAX_DOCUMENTS}篇文档')
    cleaned=[];seen=set()
    for index,document in enumerate(documents,1):
        if not isinstance(document,dict): raise ValueError(f'第{index}篇补访文档不是对象')
        doc_id=document.get('doc_id'); text=document.get('text')
        if not isinstance(doc_id,str) or not doc_id.strip(): raise ValueError(f'第{index}篇补访文档doc_id为空')
        if len(doc_id.strip())>MAX_DOC_ID: raise ValueError(f'第{index}篇补访文档doc_id超过{MAX_DOC_ID}字')
        if not isinstance(text,str): raise ValueError(f'补访文档{doc_id.strip()}的文本不是字符串')
        doc_id=doc_id.strip(); text=text.strip()
        if not 1<=len(text)<=MAX_CHARACTERS: raise ValueError(f'补访文档{doc_id}文本不在1—{MAX_CHARACTERS}字之间')
        if doc_id in seen: raise ValueError(f'补访文档doc_id重复：{doc_id}')
        if doc_id in original_ids: raise ValueError(f'补访文档doc_id与原运行文档冲突：{doc_id}')
        seen.add(doc_id);cleaned.append({'doc_id':doc_id,'text':text})
    return cleaned


def _project_vector(vector,h,hht):
    topic_count=h.shape[0]
    # 仅由当前文档向量决定的初始化，避免批大小/批次顺序影响结果。
    weights=np.full(topic_count,math.sqrt(max(float(vector.mean()),EPSILON)/topic_count),dtype=float)
    target=vector@h.T
    previous=float(np.linalg.norm(vector-weights@h))
    converged=False;iteration=0;relative_change=None;error=previous
    for iteration in range(1,PROJECTION_MAX_ITER+1):
        weights*=target/(weights@hht+EPSILON)
        error=float(np.linalg.norm(vector-weights@h))
        relative_change=abs(previous-error)/max(previous,EPSILON)
        if iteration>=PROJECTION_MIN_ITER and relative_change<PROJECTION_TOLERANCE:
            converged=True;break
        previous=error
    if not np.isfinite(weights).all() or np.any(weights<0): raise ValueError('投影权重出现非有限或负值，已拒绝返回')
    return weights,iteration,converged,relative_change,error


def project(run,documents=None,csv_text=None):
    if (documents is None)==(csv_text is None): raise ValueError('请提供documents数组或csv_text，二选一')
    if documents is not None: documents=[dict(item) if isinstance(item,dict) else {'doc_id':getattr(item,'doc_id',None),'text':getattr(item,'text',None)} for item in documents]
    basis=build_basis(run)
    if csv_text is not None: documents=parse_csv(csv_text)
    documents=validate_documents(documents,basis['original_all_ids'])

    lexicon=basis['lexicon'];vocabulary=basis['vocabulary'];idf=basis['idf'];h=basis['h']
    position={term:index for index,term in enumerate(vocabulary)}
    hht=h@h.T;topic_count=h.shape[0];topic_ids=basis['topic_ids']
    projected=[];isolated=[];vector_norms=[]
    for document in documents:
        tokens=tokenize(document['text'],lexicon['terms'],lexicon['stopwords'])
        doc_id=document['doc_id']
        if not tokens['tokens']:
            isolated.append({'doc_id':doc_id,'reason':'无有效受控词项（空词向量）','token_count':0,
                             'unknown_characters':tokens['unknown_characters'],'out_of_vocabulary_terms':[]})
            continue
        vector=np.zeros(len(vocabulary),dtype=float);outside=set()
        for term,count in tokens['counts'].items():
            index=position.get(term)
            if index is None: outside.add(term)
            else: vector[index]=count/len(tokens['tokens'])*idf[index]
        if not np.any(vector>0):
            isolated.append({'doc_id':doc_id,'reason':'词项均不在原运行特征词表（特征向量全零，无法投影）',
                             'token_count':len(tokens['tokens']),'unknown_characters':tokens['unknown_characters'],
                             'out_of_vocabulary_terms':sorted(outside)})
            continue
        weights,iterations,converged,relative_change,error=_project_vector(vector,h,hht)
        norm=float(np.linalg.norm(vector));vector_norms.append(norm)
        similarity=np.clip(basis['original_matrix']@vector/(basis['original_norms']*norm),0,1)
        neighbors=sorted(range(len(basis['original_ids'])),key=lambda i:(-round(float(similarity[i]),12),basis['original_ids'][i]))[:NEIGHBOR_LIMIT]
        total=float(weights.sum())
        projected.append({'doc_id':doc_id,'token_count':len(tokens['tokens']),'unknown_characters':tokens['unknown_characters'],
            'out_of_vocabulary_terms':sorted(outside),'weights':[float(x) for x in weights],
            'display_fractions':[float(x/total) for x in weights] if total else [0.]*topic_count,
            'iterations':iterations,'converged':converged,'relative_change':relative_change,
            'reconstruction_error':error,'relative_reconstruction_error':error/norm,
            'neighbors':[{'doc_id':basis['original_ids'][i],'similarity':float(similarity[i])} for i in neighbors]})

    total_error=math.sqrt(sum(item['reconstruction_error']**2 for item in projected))
    total_norm=math.sqrt(sum(norm**2 for norm in vector_norms))
    summary={'input_documents':len(documents),'included':len(projected),'isolated':len(isolated),
        'aggregate_reconstruction_error':total_error,
        'aggregate_relative_reconstruction_error':total_error/total_norm if total_norm else 0.,
        'max_relative_reconstruction_error':max((item['relative_reconstruction_error'] for item in projected),default=0.),
        'topic_count':topic_count,'neighbor_limit':NEIGHBOR_LIMIT}
    basis_info={'lexicon':{'id':lexicon.get('id'),'name':lexicon.get('name'),'version':lexicon.get('version'),
                    'term_count':len(lexicon['terms']),'stopword_count':len(lexicon['stopwords'])},
        'topic_count':topic_count,'topic_ids':topic_ids,'vocabulary_size':len(vocabulary),
        'vocabulary_checksum':hashlib.sha256(_stable(vocabulary).encode()).hexdigest(),
        'idf_source':'原运行持久化idf，未重算','vocabulary_order':'原运行持久化顺序，未重排',
        'original_document_count':len(basis['original_ids']),'basis_checksum':basis['checksum']}
    return {'run_id':run.get('id'),'status':run.get('status'),'basis':basis_info,'documents':projected,'isolated':isolated,
        'summary':summary,
        'projection':{'method':'固定H的非负乘法更新（只解W，不更新H）','max_iterations':PROJECTION_MAX_ITER,
            'minimum_iterations':PROJECTION_MIN_ITER,'tolerance':PROJECTION_TOLERANCE,
            'initialization':'每篇文档独立使用sqrt(mean(v)/K)初始化','neighbor_limit':NEIGHBOR_LIMIT},
        'limitations':[
            '投影沿用原运行的词典快照、词项顺序、idf与H；不重新训练、不重算idf、不改动原运行或人工命名',
            '同一补访文档单独提交、拆批或换序，权重/误差/近邻均只由该文档与原运行基底决定，结果相同',
            'TF分母为该文档全部有效受控词项；原运行词表外的词项计入分母但不进入特征，并单独记录',
            '非负权重为固定主题基底的近似投影，重建误差按||v−wH||计算；不证明语义分类或全局最优',
            '近邻只与原运行有效文档比较（行L2归一余弦，取前3，同值按doc_id排序），不包含补访文档之间',
            '无有效词项或特征向量全零的文档单独列出原因，不进入权重、误差与近邻计算']}
