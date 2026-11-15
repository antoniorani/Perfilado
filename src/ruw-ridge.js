/** Browser inference for the genuinely trained 9-class RUW Ridge classifier.
 * Training is performed with scikit-learn offline from the original RUW TRAIN split.
 * Only classifier coefficients/vocabulary are distributed, never original tweets.
 */
import { RUW_REVISION } from "./ruw-data.js";

const SOURCE="./data/ruw-joint-ridge-v1.json";
const LABELS=["AGAINST","FAVOR","NONE"];
let modelPromise;
const normalize=(text)=>String(text).normalize("NFKC").toLowerCase();

function wordNgrams(text, limit=3) {
  const words=normalize(text).match(/[\p{L}\p{N}_]{2,}/gu)||[];
  const tokens=[];
  for(let n=1;n<=limit;n++){
    for(let i=0;i+n<=words.length;i++)tokens.push(words.slice(i,i+n).join(" "));
  }
  return tokens;
}

/** Equivalent to scikit-learn TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5)). */
function charBoundaryNgrams(text, first=3, last=5) {
  const content=normalize(text).replace(/\s+/g," ");
  const tokens=[];
  for(const word of content.split(" ").filter(Boolean)){
    const padded=` ${word} `;
    for(let n=first;n<=last;n++){
      let emitted=false;
      for(let i=0;i+n<=padded.length;i++){
        tokens.push(padded.slice(i,i+n));
        emitted=true;
      }
      if(!emitted)tokens.push(padded);
    }
  }
  return tokens;
}

function buildExtractor(config) {
  if(!["word","char_wb"].includes(config.analyzer)
      || !Array.isArray(config.vocabulary) || !Array.isArray(config.idf)
      || config.vocabulary.length!==config.idf.length
      || ![1,.5].includes(config.scale)) {
    throw new Error("Representación TF-IDF preentrenada incompatible");
  }
  const index=new Map(config.vocabulary.map((term,i)=>[term,i]));
  const idf=Float32Array.from(config.idf);
  if(idf.some(x=>!Number.isFinite(x)))throw new Error("IDF inválido");
  return {
    size:idf.length, analyzer:config.analyzer, terms:config.vocabulary,
    scale:config.scale,
    transform(text) {
      const counts=new Map();
      const tokens=config.analyzer==="word"?wordNgrams(text,3):charBoundaryNgrams(text);
      for(const token of tokens){
        const i=index.get(token);
        if(i!==undefined)counts.set(i,(counts.get(i)||0)+1);
      }
      let norm=0;
      const pairs=[];
      for(const [i,count] of counts) {
        const weighted=(1+Math.log(count))*idf[i];
        pairs.push([i,weighted]);
        norm+=weighted*weighted;
      }
      norm=Math.sqrt(norm)||1;
      return pairs.map(([i,value])=>[i,value/norm*config.scale]);
    },
  };
}

export function createRidgeModel(payload,target) {
  if(payload?.format!=="RUW-joint-ridge-1" || payload.revision!==RUW_REVISION
    || !["russia","ukraine"].includes(target)
    || payload.classNames?.length!==9 || payload.representation?.length!==2
    || payload.weightsI16?.scale?.length!==9 || payload.intercepts?.length!==9) {
    throw new Error("Modelo RUW preentrenado incompatible con la versión del corpus");
  }
  const extractors=payload.representation.map(buildExtractor);
  const size=extractors.reduce((s,x)=>s+x.size,0);
  const {weightsI16}=payload;
  if(weightsI16.shape?.[0]!==9 || weightsI16.shape?.[1]!==size) {
    throw new Error("Dimensiones de pesos incompatibles");
  }
  const bytes=atob(weightsI16.bytes);
  if(bytes.length!==9*size*2)throw new Error("Archivo de pesos incompleto");
  const raw=new Uint8Array(bytes.length);
  for(let i=0;i<bytes.length;i++)raw[i]=bytes.charCodeAt(i);
  const quantized=new Int16Array(raw.buffer);
  const coeffs=weightsI16.scale.map((scale,c)=>{
    if(!Number.isFinite(scale)||scale<=0)throw new Error("Escala del modelo inválida");
    return Float32Array.from(quantized.subarray(c*size,(c+1)*size),v=>v*scale);
  });
  const intercepts=Float32Array.from(payload.intercepts);
  const classNames=payload.classNames.map(name=>name.split("/"));
  if(classNames.some((pair)=>pair.length!==2 || pair.some(label=>!LABELS.includes(label)))) {
    throw new Error("Etiquetas conjuntas inválidas en el modelo");
  }
  const targetIndex=target==="russia"?0:1;
  return {
    name:"joint-ridge",target,vectorizer:{size},
    provenance:payload.devValidation,
    predictText(text) {
      const scores=Array.from(intercepts);
      let knownTerms=0, offset=0;
      for(const extractor of extractors) {
        const vector=extractor.transform(text);
        knownTerms+=vector.length;
        for(const [term,value] of vector) {
          const pos=offset+term;
          for(let c=0;c<scores.length;c++)scores[c]+=coeffs[c][pos]*value;
        }
        offset+=extractor.size;
      }
      let best=0;
      for(let c=1;c<scores.length;c++)if(scores[c]>scores[best])best=c;
      return {
        label:classNames[best][targetIndex],
        jointLabels:{russia:classNames[best][0],ukraine:classNames[best][1]},
        scores, knownTerms,
      };
    },
    topTerms(label,count=6) {
      if(!LABELS.includes(label))throw new Error("Etiqueta desconocida");
      const wordExtractor=extractors[0];
      const values=[];
      for(let i=0;i<wordExtractor.size;i++) {
        let positives=0,negative=0,np=0,nn=0;
        for(let c=0;c<coeffs.length;c++) {
          if(classNames[c][targetIndex]===label) {positives+=coeffs[c][i];np++;}
          else {negative+=coeffs[c][i];nn++;}
        }
        values.push({term:wordExtractor.terms[i],weight:positives/np-negative/nn});
      }
      values.sort((a,b)=>b.weight-a.weight);
      return values.slice(0,count);
    },
  };
}

export async function loadRidgeModel(target,fetcher=fetch) {
  if(!modelPromise)modelPromise=(async()=>{
    const response=await fetcher(SOURCE);
    if(!response.ok)throw new Error(`No se pudo cargar el modelo RUW (HTTP ${response.status})`);
    return response.json();
  })().catch(error=>{modelPromise=null;throw error;});
  return createRidgeModel(await modelPromise,target);
}
