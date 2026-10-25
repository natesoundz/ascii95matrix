#!/usr/bin/env python3
"""Conical 437 / Webster SQL-IR transformer, executable deterministic prototype.
Real workbook-derived JSON required. Real SQLite required. No synthetic data.
Six relational channels compile into factored 437-dimensional projections.
"""
import argparse, hashlib, json, sqlite3, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

D=437
def digest(x):
    return hashlib.sha256(str(x).encode("utf-8")).digest()
# Direct, typed workbook feature activation. No hash-based numeric imputation.
# Deferred/status cells remain inactive, and their status is retained in raw JSON.
UNRESOLVED={"X","E","R","M","U","V","K","S","Na","none",""}
def encode_cell(v):
    if v is None: return 0.
    t=str(v).strip()
    if t in UNRESOLVED or t=="0": return 0.
    if t=="1": return 1.
    if isinstance(v,(int,float)) and not isinstance(v,bool):
        return 1. if float(v)>0 else 0.
    # Actual populated categorical cells are known evidence of that attribute,
    # not hashed numbers or invented magnitudes.
    try:
        return 1. if float(t)>0 else 0.
    except ValueError:
        return 1.

def load_workbook(path):
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    assert len(raw["attribute_ids"])==D
    assert len(set(raw["attribute_ids"]))==D
    records=raw["records"]
    assert len(records)==95
    assert [r["ascii"] for r in records]==list(range(32,127))
    E=np.array([[encode_cell(v) for v in r["cells"]] for r in records],dtype=np.float64)
    assert E.shape==(95,D) and np.isfinite(E).all() and np.min(E)>=0
    return E,raw

def direct_operator(E,T):
    """C_ij=sum_ab T_ab F_ai F_bj; B_ij=C_ij/sum_j C_ij.
    F is an explicit, nonnegative attribute activation from the real workbook.
    All-zero evidence rows receive identity. Exactly row stochastic.
    """
    C=E.T @ T @ E
    C=np.maximum(C,0.)
    row_sums=C.sum(axis=1)
    B=np.zeros_like(C)
    good=row_sums>0
    B[good]=C[good]/row_sums[good,None]
    bad=np.flatnonzero(~good)
    B[bad,bad]=1.
    assert np.isfinite(B).all() and B.min()>=0.
    assert np.allclose(B.sum(axis=1),1,rtol=0,atol=1e-10)
    return B

def char_hist(s):
    z=np.zeros(95,dtype=np.float64)
    for c in str(s):
        k=ord(c)-32
        if 0<=k<95: z[k]+=1.
    if z.sum(): z/=z.sum()
    return z

def add_pair(T,a,b,w=1.):
    if not a or not b: return
    aa=char_hist(a);bb=char_hist(b)
    if aa.sum() and bb.sum() and np.isfinite(w) and w>0:
        ia=np.flatnonzero(aa); ib=np.flatnonzero(bb)
        T[np.ix_(ia,ib)]+=float(w)*np.outer(aa[ia],bb[ib])

# These SQL statements use actual fields from the 33-table Webster schema.
QUERIES={
 "Q":"SELECT normalized FROM lexeme ORDER BY lexeme_id",
 "K":"SELECT left_lemma,right_lemma,frequency FROM collocation ORDER BY collocation_id",
 "V":"""SELECT l.normalized,r.target_lemma
         FROM relation r JOIN lexeme l ON l.lexeme_id=r.source_lexeme_id
         WHERE r.target_lemma IS NOT NULL ORDER BY r.relation_id""",
 "O":"""SELECT a.normalized,b.normalized,d.term_count
         FROM definition_term d JOIN webster_entry a ON a.webster_id=d.webster_id
         JOIN webster_entry b ON b.webster_id=d.term_webster_id
         ORDER BY d.webster_id,d.term_webster_id""",
 "F1":"""SELECT o.lemma,h.lemma
          FROM occurrence o JOIN occurrence h
          ON h.example_id=o.example_id AND h.token_index=o.head_index
          WHERE o.head_index IS NOT NULL ORDER BY o.occurrence_id""",
 "F2":"SELECT l.normalized,i.normalized FROM inflection_form i JOIN lexeme l ON l.lexeme_id=i.lexeme_id ORDER BY i.inflection_id"
}
def compile_channel(conn,name,limit=0):
    T=np.zeros((95,95),dtype=np.float64)
    sql=QUERIES[name]+(" LIMIT ?" if limit else "")
    cur=conn.execute(sql,(limit,) if limit else ())
    count=0
    for row in cur:
        count+=1
        if name=="Q":
            s=row[0] or ""
            for a,b in zip(s,s[1:]): add_pair(T,a,b)
        elif name in ("K","O"):
            add_pair(T,row[0],row[1],max(0.,float(row[2] or 0)))
        else:
            add_pair(T,row[0],row[1])
    # Preserve zero-evidence rows via identity. Normalize counts by source row.
    sums=T.sum(axis=1)
    for i in range(95):
        if sums[i]>0: T[i]/=sums[i]
        else: T[i,i]=1.
    return T,count

def norm(x):
    return (x-x.mean(axis=-1,keepdims=True))/np.sqrt(x.var(axis=-1,keepdims=True)+1e-5)
def relu(x): return np.maximum(x,0.)
def softmax_causal(scores):
    scores=np.tril(scores)+np.triu(np.full_like(scores,-np.inf),1)
    maxima=np.max(scores,axis=1,keepdims=True)
    z=np.exp(scores-maxima)
    z[~np.isfinite(z)]=0
    return z/z.sum(axis=1,keepdims=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workbook",required=True,help="Extracted real workbook JSON")
    p.add_argument("--db",required=True,help="Real webster_full_lexicon.sqlite")
    p.add_argument("--text",required=True,help="ASCII95 input")
    p.add_argument("--limit",type=int,default=0,help="Optional rows/channel; 0 means ALL")
    p.add_argument("--output",default="conical_sql_ir_result.json")
    a=p.parse_args()
    E,raw=load_workbook(a.workbook)
    if any(ord(c)<32 or ord(c)>126 for c in a.text):
        p.error("Only ASCII 32..126 is supported by the workbook.")
    if not a.text:p.error("Nonempty input required")
    uri=Path(a.db).resolve().as_uri()+"?mode=ro"
    con=sqlite3.connect(uri,uri=True)
    con.execute("PRAGMA query_only=ON")
    required={"lexeme","collocation","relation","definition_term","webster_entry","occurrence","inflection_form"}
    actual={x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not required.issubset(actual):
        raise RuntimeError("Missing actual schema tables: "+str(sorted(required-actual)))
    started=time.time()
    T={};counts={}
    for name in QUERIES:
        T[name],counts[name]=compile_channel(con,name,a.limit)
        print("COMPILED",name,"rows",counts[name],flush=True)
    con.close()
    # Direct 437-coordinate count construction, no least-squares lift.
    # B_pi is explicitly nonnegative and row stochastic for every channel.
    B={name:direct_operator(E,T[name]) for name in QUERIES}
    def project(x,name):
        return x@B[name]
    X=E[np.array([ord(c)-32 for c in a.text])]
    Z1=norm(X)
    Q=project(Z1,"Q");K=project(Z1,"K");V=project(Z1,"V")
    A=softmax_causal((Q@K.T)/np.sqrt(D))
    H=A@V
    O=project(H,"O")
    Z2=X+O
    Z3=norm(Z2)
    U=relu(project(Z3,"F1"))
    delta=project(U,"F2")
    Y=Z2+delta
    assert Y.shape==X.shape and np.isfinite(Y).all()
    # Nearest workbook vector is a diagnostic only, NOT a validated decoder.
    distance=((Y[:,None,:]-E[None,:,:])**2).sum(axis=2)
    nearest="".join(chr(32+int(i)) for i in distance.argmin(axis=1))
    out={
        "status":"EXECUTED","input":a.text,"shape":list(Y.shape),
        "compiled_rows":counts,"query_limit":a.limit,
        "operator_sha256":{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in B.items()},
        "operator_invariants":{k:{"min":float(v.min()),"max_row_sum_error":float(np.max(np.abs(v.sum(axis=1)-1)))} for k,v in B.items()},
        "input_vectors":X.tolist(),"output_vectors":Y.tolist(),
        "attention":A.tolist(),"nearest_workbook_char_diagnostic":nearest,
        "elapsed_seconds":round(time.time()-started,3),
        "disclosures":[
            "This is a deterministic SQL-derived projection prototype, not an equivalent trained transformer.",
            "Typed workbook unresolved cells are retained in source JSON and numerically inactive; known categorical cells activate their own attribute column.",
            "SQL rows map through character histograms and explicit workbook column activation: C_ij=sum_ab T_ab E_ai E_bj; B=row_normalize(C).",
            "Character-histogram correspondence is a deterministic engineering choice, not proof of linguistic semantics.",
            "Nearest workbook character is not a validated language decoder.",
            "Full database results used unless --limit is explicitly supplied."
        ]}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("DONE",a.output,"shape",Y.shape,"nearest_diagnostic",repr(nearest))
if __name__=="__main__":main()
