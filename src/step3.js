import { loadRUW, LABELS, LABEL_NAMES, TARGETS, labelCounts, normalizedText, splitTrainValidation } from "./ruw-data.js";
import {
  MODEL_NAMES, buildVectorizer, trainClassifier, evaluateClassifier, evaluatePredictions, baselineLabel,
} from "./text-classifier.js";
import { setupAnswers } from "./answers.js";


const $ = (selector) => document.querySelector(selector);
const number = new Intl.NumberFormat("es-ES");
const percent = (value) => value == null ? "—" : `${new Intl.NumberFormat("es-ES", {
  style:"percent", maximumFractionDigits:1,
}).format(value)}`;
const PER_PAGE = 5;

const controls = {
  status:$("#ruw-status"), loadTitle:$("#load-title"), loadDescription:$("#load-description"),
  content:$("#corpus-content"), train:$("#train-model"),
  trainStatus:$("#train-status"), trainDescription:$("#train-description"),
  trainFeedback:$("#train-feedback"), results:$("#validation-results"),
  target:$("#target-select"), models:document.querySelectorAll('input[name="model"]'),
  ngrams:$("#ngram-select"), vocab:$("#vocab-select"), alpha:$("#alpha-select"),
  split:$("#split-range"), splitValue:$("#split-value"), splitTrainBar:$("#split-train-bar"),
  splitValBar:$("#split-val-bar"), splitTrainCount:$("#split-train-count"), splitValCount:$("#split-val-count"),
  exploreTarget:$("#explore-target"), exploreLabel:$("#explore-label"), exploreQuery:$("#explore-query"),
  browse:$("#tweet-samples"), browseSummary:$("#explore-summary"), browseCount:$("#explore-counter"),
  browsePrev:$("#explore-prev"), browseNext:$("#explore-next"),
  errorFilter:$("#error-filter"), errorList:$("#error-examples"), errorSummary:$("#error-summary"),
  errorCount:$("#error-counter"), errorPrev:$("#error-prev"), errorNext:$("#error-next"),
  manual:$("#manual-text"), manualButton:$("#manual-run"), manualResult:$("#manual-result"),
  summary:$("#train-summary"),
};

let corpus=null;
let rows=[];
let split=null;
let model=null;
let validation=null;
let selectedTarget="russia";
let exploreRows=[];
let explorePage=1;
let reviewRows=[];
let reviewPage=1;
let configurationVersion=0;
let semanticWorker=null;
let semanticRequestId=0;
const semanticRequests=new Map();

function runSemantic(type, payload, onProgress=()=>{}) {
  if(!semanticWorker) {
    semanticWorker=new Worker(new URL("./semantic-worker.js", import.meta.url),{type:"module"});
    semanticWorker.addEventListener("message",({data})=>{
      const request=semanticRequests.get(data.id);
      if(!request)return;
      if(data.type==="progress"){request.onProgress(data.message);return;}
      semanticRequests.delete(data.id);
      if(data.type==="error")request.reject(new Error(data.message));
      else if(data.type==="result")request.resolve(data.result);
      else request.reject(new Error("Respuesta desconocida del trabajador"));
    });
    semanticWorker.addEventListener("error",(event)=>{
      for(const request of semanticRequests.values())request.reject(new Error(event.message||"No se pudo cargar el motor de embeddings"));
      semanticRequests.clear();
      semanticWorker.terminate();
      semanticWorker=null;
    });
  }
  const id=++semanticRequestId;
  return new Promise((resolve,reject)=>{
    semanticRequests.set(id,{resolve,reject,onProgress});
    semanticWorker.postMessage({id,type,...payload});
  });
}

function element(tag, className, content) {
  const node=document.createElement(tag);
  if(className)node.className=className;
  if(content!==undefined)node.textContent=String(content);
  return node;
}
function makeChip(label) {
  return element("span",`stance-chip stance-${label.toLowerCase()}`,LABEL_NAMES[label]);
}
function emptyPanel(text) { return element("p","text-empty",text); }

function drawBars(container, counts) {
  const total=Object.values(counts).reduce((a,b)=>a+b,0);
  const max=Math.max(...Object.values(counts),1);
  const fragment=document.createDocumentFragment();
  for(const label of LABELS) {
    const row=element("div","chart-line");
    const name=element("span","chart-line-label",LABEL_NAMES[label]);
    const track=element("span","chart-track");
    const fill=element("span",`chart-fill stance-fill-${label.toLowerCase()}`);
    fill.style.display="block";
    fill.style.width=`${100*counts[label]/max}%`;
    track.append(fill);
    const value=element("span","chart-line-count",`${number.format(counts[label])} (${percent(counts[label]/total)})`);
    row.append(name,track,value);
    fragment.append(row);
  }
  container.replaceChildren(fragment);
}
function renderJointLabels() {
  const counts=Object.fromEntries(LABELS.map(a=>[a,Object.fromEntries(LABELS.map(b=>[b,0]))]));
  for(const row of rows) counts[row.russia][row.ukraine] += 1;
  const table=element("table","ruw-matrix joint-matrix");
  const head=element("thead");
  const top=element("tr");
  top.append(element("th","ruw-matrix-corner","Rusia ↓ / Ucrania →"));
  for(const label of LABELS) top.append(element("th","",LABEL_NAMES[label]));
  head.append(top);
  const body=element("tbody");
  for(const r of LABELS) {
    const line=element("tr");
    const heading=element("th","",LABEL_NAMES[r]);heading.scope="row";
    line.append(heading);
    for(const u of LABELS) line.append(element("td","",number.format(counts[r][u])));
    body.append(line);
  }
  table.append(head,body);
  $("#joint-labels").replaceChildren(table);
}

function renderCorpus() {
  const sum=(counts)=>Object.values(counts).reduce((total,n)=>total+n,0);
  $("#count-original").textContent=number.format(sum(corpus.originalCounts));
  $("#count-removed").textContent=number.format(sum(corpus.removed));
  $("#count-total").textContent=number.format(rows.length);
  drawBars($("#russia-bars"),labelCounts(rows,"russia"));
  drawBars($("#ukraine-bars"),labelCounts(rows,"ukraine"));
  renderJointLabels();
}

function tweetCard(row, { truth=false, predicted=null, selected=null }={}) {
  const card=element("article","tweet-card");
  const header=element("div","tweet-card-head");
  header.append(element("span","text-code",row.id));
  if(truth) {
    const result=element("span",predicted===row[selected]?"text-outcome correct":"text-outcome wrong",
      predicted===row[selected]?"Correcto":"Clasificación equivocada");
    header.append(result);
  }
  const body=element("p","tweet-body",row.text || "(Texto vacío tras retirar identificadores)");
  const labels=element("div","tweet-labels");
  if(truth) {
    labels.append(element("span","tweet-label-name",`Etiqueta real respecto a ${TARGETS[selected]}:`),
      makeChip(row[selected]),element("span","tweet-label-name","Predicción:"),makeChip(predicted));
  } else {
    labels.append(element("span","tweet-label-name","Rusia:"),makeChip(row.russia),
      element("span","tweet-label-name","Ucrania:"),makeChip(row.ukraine));
  }
  card.append(header,body,labels);
  return card;
}

function updateExploreFilter() {
  const term=normalizedText(controls.exploreQuery.value);
  const target=controls.exploreTarget.value,label=controls.exploreLabel.value;
  exploreRows=rows.filter((row)=>
    (label==="all" || row[target]===label) && (!term || normalizedText(row.text).includes(term)));
  explorePage=1;
  renderExplore();
}
function renderExplore() {
  const start=(explorePage-1)*PER_PAGE;
  const rows=exploreRows.slice(start,start+PER_PAGE);
  const fragment=document.createDocumentFragment();
  if(!rows.length) fragment.append(emptyPanel("No hay publicaciones que coincidan con los filtros."));
  else for(const row of rows) fragment.append(tweetCard(row));
  controls.browse.replaceChildren(fragment);
  const pages=Math.max(1,Math.ceil(exploreRows.length/PER_PAGE));
  controls.browseSummary.textContent=exploreRows.length
    ? `Mostrando ${start+1}–${Math.min(start+PER_PAGE,exploreRows.length)} de ${number.format(exploreRows.length)} textos`
    : "Sin resultados";
  controls.browseCount.textContent=`${explorePage} / ${pages}`;
  controls.browsePrev.disabled=explorePage<=1;
  controls.browseNext.disabled=explorePage>=pages;
}

function classConfusion(mount,stats) {
  const table=element("table","ruw-matrix confusion-table");
  const head=element("thead"),headRow=element("tr");
  headRow.append(element("th","ruw-matrix-corner","Real ↓ / Predicho →"));
  for(const label of LABELS)headRow.append(element("th","",LABEL_NAMES[label]));
  head.append(headRow);
  const body=element("tbody");
  const maximum=Math.max(...stats.matrix.flat(),1);
  for(let i=0;i<LABELS.length;i++){
    const row=element("tr");
    const heading=element("th","",LABEL_NAMES[LABELS[i]]);heading.scope="row";
    row.append(heading);
    for(let j=0;j<LABELS.length;j++){
      const cell=element("td",i===j?"confusion-right":"confusion-wrong",number.format(stats.matrix[i][j]));
      cell.style.setProperty("--heat-opacity",String(.08 + .34*stats.matrix[i][j]/maximum));
      row.append(cell);
    }
    body.append(row);
  }
  table.append(head,body);
  mount.replaceChildren(table);
}

function classMetrics(mount,stats) {
  const table=element("table","text-metric-table");
  const head=element("thead"),tr=element("tr");
  for(const title of ["Clase","Precisión","Sensibilidad","F1","N"])tr.append(element("th","",title));
  head.append(tr);
  const body=element("tbody");
  for(const entry of stats.perClass){
    const row=element("tr");
    row.append(element("th","",LABEL_NAMES[entry.label]),
      element("td","",percent(entry.precision)),element("td","",percent(entry.recall)),
      element("td","",percent(entry.f1)),element("td","",number.format(entry.support)));
    body.append(row);
  }
  table.append(head,body);
  mount.replaceChildren(table);
}

function featureTerms() {
  if(model.name==="semantic"){
    $("#feature-terms").replaceChildren(element("p","",
      "MiniLM transforma cada publicación en 384 componentes semánticos. La cabeza logística aprende pesos sobre esos componentes; no dispone de una lista fiable de palabras que expliquen una predicción."));
    return;
  }
  const fragment=document.createDocumentFragment();
  for(const label of LABELS){
    const group=element("div","feature-group");
    const heading=element("h4","",LABEL_NAMES[label]);
    const tags=element("div","term-tags");
    for(const item of model.topTerms(label,7)){
      tags.append(element("span","term-tag",item.term.replaceAll("·"," + ")));
    }
    group.append(heading,tags);
    fragment.append(group);
  }
  $("#feature-terms").replaceChildren(fragment);
}

function updateReviews() {
  const filter=controls.errorFilter.value;
  reviewRows=validation.decisions.filter((entry)=>
    filter==="all" || (filter==="errors" ? !entry.correct : entry.correct));
  reviewPage=1;
  renderReviews();
}
function renderReviews() {
  const start=(reviewPage-1)*PER_PAGE;
  const fragment=document.createDocumentFragment();
  const slice=reviewRows.slice(start,start+PER_PAGE);
  if(!slice.length)fragment.append(emptyPanel("No hay casos de este tipo con esta configuración."));
  else for(const item of slice) fragment.append(tweetCard(item.row,
    {truth:true,predicted:item.prediction,selected:selectedTarget}));
  controls.errorList.replaceChildren(fragment);
  const pages=Math.max(1,Math.ceil(reviewRows.length/PER_PAGE));
  controls.errorSummary.textContent=`${number.format(reviewRows.length)} de ${number.format(validation.total)} resultados`;
  controls.errorCount.textContent=`${reviewPage} / ${pages}`;
  controls.errorPrev.disabled=reviewPage<=1;
  controls.errorNext.disabled=reviewPage>=pages;
}

function showValidation() {
  const majority=baselineLabel(split.train,selectedTarget);
  const baseline=split.val.filter(row=>row[selectedTarget]===majority).length/split.val.length;
  $("#metric-accuracy").textContent=percent(validation.accuracy);
  $("#metric-macro").textContent=percent(validation.macroF1);
  $("#metric-baseline").textContent=percent(baseline);
  $("#metric-vocab").textContent=number.format(model.vectorizer.size);
  classConfusion($("#confusion-matrix"),validation);
  classMetrics($("#class-metrics"),validation);
  featureTerms();
  updateReviews();
  controls.results.hidden=false;
  controls.manualResult.textContent="";
}

const selectedModel=()=>[...controls.models].find(input=>input.checked).value;

function renderSplit() {
  const share=Number(controls.split.value);
  controls.splitValue.textContent=`${share} %`;
  controls.splitTrainBar.style.flexGrow=share;
  controls.splitValBar.style.flexGrow=100-share;
  split=rows.length ? splitTrainValidation(rows,share/100) : null;
  controls.splitTrainCount.textContent=split ? number.format(split.train.length) : "—";
  controls.splitValCount.textContent=split ? number.format(split.val.length) : "—";
}

function configurationChanged() {
  configurationVersion++;
  model=null;
  validation=null;
  controls.results.hidden=true;
  const chosenModel=selectedModel();
  for(const node of document.querySelectorAll("[data-models]")) {
    node.hidden=!node.dataset.models.split(" ").includes(chosenModel);
  }
  renderSplit();
  controls.summary.textContent=split
    ? `Entrenarás «${MODEL_NAMES[chosenModel]}» para predecir la postura hacia ${TARGETS[controls.target.value]} con ${number.format(split.train.length)} textos, y lo evaluarás con ${number.format(split.val.length)} textos de validación que no verá durante el entrenamiento.`
    : "Cuando se cargue el corpus podrás entrenar esta configuración.";
  if(!corpus)return;
  controls.trainStatus.textContent="Configuración lista";
  controls.trainDescription.textContent="Pulsa «Entrenar y validar» para evaluar esta configuración.";
}

async function train() {
  if(!corpus)return;
  controls.train.disabled=true;
  const version=configurationVersion;
  const chosenModel=selectedModel();
  const {train:trainRows,val:valRows}=split;
  const target=controls.target.value;
  controls.trainStatus.textContent=chosenModel==="semantic"
    ? "Calculando embeddings y entrenando en el navegador…"
    : "Entrenando el clasificador TF-IDF en el navegador…";
  controls.trainDescription.textContent="No se envían publicaciones a servicios de inferencia. La representación y los pesos se calculan localmente.";
  await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  try {
    let nextModel,nextValidation;
    if(chosenModel==="semantic") {
      const outcome=await runSemantic("train",{
        target,
        train:trainRows.map(row=>({text:row.text,label:row[target]})),
        val:valRows.map(row=>row.text),
      },message=>{if(version===configurationVersion)controls.trainDescription.textContent=message;});
      nextModel={name:"semantic",target,vectorizer:{size:outcome.dimension}};
      nextValidation=evaluatePredictions(valRows,target,outcome.predictions);
    } else {
      const vectorizer=buildVectorizer(trainRows,
        {ngrams:Number(controls.ngrams.value),maxFeatures:Number(controls.vocab.value)});
      nextModel=trainClassifier(trainRows,target,chosenModel,vectorizer,
        {alpha:Number(controls.alpha.value)});
      nextValidation=evaluateClassifier(nextModel,valRows,target);
    }
    if(version!==configurationVersion)return;
    selectedTarget=target;
    model=nextModel;
    validation=nextValidation;
    controls.trainStatus.textContent="Modelo entrenado: "+MODEL_NAMES[model.name];
    controls.trainDescription.textContent=
      number.format(trainRows.length)+" textos de entrenamiento y "+number.format(valRows.length)+
      " de validación. "+number.format(model.vectorizer.size)+" características. F1 macro en validación: "+
      percent(validation.macroF1)+".";
    showValidation();
    controls.results.scrollIntoView({behavior:"smooth",block:"start"});
  } catch(error) {
    if(version!==configurationVersion)return;
    console.error("Error de entrenamiento RUW:",error);
    controls.trainStatus.textContent="No se pudo entrenar el clasificador";
    controls.trainDescription.textContent=error.message + (chosenModel==="semantic"
      ? ". La primera carga del encoder requiere acceso a cdn.jsdelivr.net y huggingface.co; comprueba la conexión y vuelve a intentarlo."
      : "");
    controls.results.hidden=true;
  } finally {
    controls.train.disabled=false;
  }
}
async function predictManual() {
  if(!model)return;
  const text=controls.manual.value.trim();
  if(!text){
    controls.manualResult.textContent="Escribe una frase antes de clasificar.";
    return;
  }
  const version=configurationVersion;
  controls.manualButton.disabled=true;
  controls.manualResult.textContent="Calculando la predicción localmente…";
  try {
    const result=model.name==="semantic"
      ? await runSemantic("manual",{target:selectedTarget,text})
      : model.predictText(text);
    if(version!==configurationVersion)return;
    controls.manualResult.textContent=
      "Predicción para "+TARGETS[selectedTarget]+": "+LABEL_NAMES[result.label]+". "+
      (model.name==="semantic"?"Basada en un embedding semántico de 384 dimensiones. ":
        "Términos conocidos por el modelo: "+result.knownTerms+". ")+
      "Es una clasificación del texto, no una conclusión sobre la persona que lo escribió.";
  } catch(error) {
    if(version===configurationVersion)controls.manualResult.textContent=error.message;
  } finally {
    controls.manualButton.disabled=false;
  }
}

async function init() {
  setupAnswers({first:13,last:19,step:3,title:"PERFILADO — PASO 3: ANÁLISIS DE TEXTOS",
    filename:"perfilado-paso3-respuestas.txt"});
  for(const field of [controls.exploreTarget,controls.exploreLabel]) {
    field.addEventListener("change",updateExploreFilter);
  }
  controls.exploreQuery.addEventListener("input",updateExploreFilter);
  controls.browsePrev.addEventListener("click",()=>{if(explorePage>1){explorePage-=1;renderExplore();}});
  controls.browseNext.addEventListener("click",()=>{if(explorePage*PER_PAGE<exploreRows.length){explorePage+=1;renderExplore();}});
  for(const field of [controls.target,...controls.models,controls.ngrams,controls.vocab,controls.alpha]) {
    field.addEventListener("change",configurationChanged);
  }
  controls.split.addEventListener("input",configurationChanged);
  controls.train.addEventListener("click",train);
  controls.manualButton.addEventListener("click",predictManual);
  controls.errorFilter.addEventListener("change",updateReviews);
  controls.errorPrev.addEventListener("click",()=>{if(reviewPage>1){reviewPage-=1;renderReviews();}});
  controls.errorNext.addEventListener("click",()=>{if(reviewPage*PER_PAGE<reviewRows.length){reviewPage+=1;renderReviews();}});
  configurationChanged();
  try{
    corpus=await loadRUW();
    rows=[...corpus.train,...corpus.val,...corpus.test];
    renderCorpus();
    controls.content.hidden=false;
    controls.train.disabled=false;
    controls.status.classList.add("text-status-success");
    controls.loadTitle.textContent="Corpus RUW cargado correctamente";
    controls.loadDescription.textContent=
      "6.000 publicaciones originales, con etiquetas de Rusia y Ucrania. El entrenamiento se ejecuta en el navegador; el modelo semántico descarga primero un encoder público.";
    configurationChanged();
    controls.trainStatus.textContent="Datos preparados para entrenar";
    controls.trainDescription.textContent="Sigue los cuatro pasos y pulsa «Entrenar y validar».";
    updateExploreFilter();
  } catch(error) {
    console.error("No se pudo obtener el corpus RUW:",error);
    controls.status.classList.add("text-status-error");
    controls.loadTitle.textContent="No se han podido cargar las publicaciones originales";
    controls.loadDescription.textContent=
      `${error.message}. Comprueba el acceso a raw.githubusercontent.com y actualiza la página. No se muestran ejemplos inventados como sustitutos.`;
    controls.trainStatus.textContent="El entrenamiento requiere el corpus original";
    controls.trainDescription.textContent="Sin los datos de RUW no se pueden generar predicciones reales.";
  }
}
init();
