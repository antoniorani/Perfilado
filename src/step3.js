import { loadRUW, LABELS, LABEL_NAMES, TARGETS, labelCounts, normalizedText } from "./ruw-data.js";
import {
  MODEL_NAMES, buildVectorizer, trainClassifier, evaluateClassifier, baselineLabel,
} from "./text-classifier.js";
import { setupAnswers } from "./answers.js";
import { loadRidgeModel } from "./ruw-ridge.js";

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
  target:$("#target-select"), model:$("#model-select"), ngrams:$("#ngram-select"),
  vocab:$("#vocab-select"), alpha:$("#alpha-select"), alphaField:$("#alpha-field"),
  exploreTarget:$("#explore-target"), exploreLabel:$("#explore-label"), exploreQuery:$("#explore-query"),
  browse:$("#tweet-samples"), browseSummary:$("#explore-summary"), browseCount:$("#explore-counter"),
  browsePrev:$("#explore-prev"), browseNext:$("#explore-next"),
  errorFilter:$("#error-filter"), errorList:$("#error-examples"), errorSummary:$("#error-summary"),
  errorCount:$("#error-counter"), errorPrev:$("#error-prev"), errorNext:$("#error-next"),
  testSection:$("#final-test"), testButton:$("#reveal-test"), testResults:$("#test-results"),
  manual:$("#manual-text"), manualButton:$("#manual-run"), manualResult:$("#manual-result"),
  modelExplanation:$("#model-explanation"),
};

let corpus=null;
let model=null;
let validation=null;
let selectedTarget="russia";
let exploreRows=[];
let explorePage=1;
let reviewRows=[];
let reviewPage=1;
let testWasViewed=false;

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
  for(const row of corpus.train) counts[row.russia][row.ukraine] += 1;
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
  $("#count-train").textContent=number.format(corpus.originalCounts.train);
  $("#count-val").textContent=number.format(corpus.originalCounts.val);
  $("#count-test").textContent=number.format(corpus.originalCounts.test);
  const excluded=Object.values(corpus.removed).reduce((sum,n)=>sum+n,0);
  $("#dedup-notice").textContent=
    `Particiones originales: 4.200 / 600 / 1.200. Se excluyeron ${excluded} publicaciones repetidas tras el filtrado de identificadores (${corpus.removed.train} del entrenamiento, ${corpus.removed.val} de validación y ${corpus.removed.test} de prueba). Quedan ${number.format(corpus.train.length)} / ${number.format(corpus.val.length)} / ${number.format(corpus.test.length)} ejemplos independientes. No se modifican las etiquetas originales.`;
  drawBars($("#russia-bars"),labelCounts(corpus.train,"russia"));
  drawBars($("#ukraine-bars"),labelCounts(corpus.train,"ukraine"));
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
  exploreRows=corpus.train.filter((row)=>
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
  const majority=baselineLabel(corpus.train,selectedTarget);
  const baseline=corpus.val.length
    ? corpus.val.filter(row=>row[selectedTarget]===majority).length/corpus.val.length : 0;
  $("#metric-accuracy").textContent=percent(validation.accuracy);
  $("#metric-macro").textContent=percent(validation.macroF1);
  $("#metric-baseline").textContent=percent(baseline);
  $("#metric-vocab").textContent=number.format(model.vectorizer.size);
  classConfusion($("#confusion-matrix"),validation);
  classMetrics($("#class-metrics"),validation);
  featureTerms();
  updateReviews();
  controls.results.hidden=false;
  controls.testSection.hidden=false;
  controls.testResults.hidden=true;
  controls.testButton.disabled=false;
  controls.manualResult.textContent="";
}

function modelChanged() {
  model=null;
  validation=null;
  controls.results.hidden=true;
  controls.testSection.hidden=true;
  controls.testResults.hidden=true;
  const advanced=controls.model.value==="joint-ridge";
  controls.alphaField.hidden=controls.model.value!=="nb";
  for(const input of [controls.ngrams,controls.vocab]) input.disabled=advanced;
  controls.train.textContent=advanced?"Cargar y validar modelo →":"Entrenar y validar →";
  controls.modelExplanation.textContent=advanced
    ? "Ridge conjunto: modelo real entrenado previamente con RUW (4.095 textos). Predice una de nueve combinaciones de postura y utiliza palabras, trigramas y fragmentos de caracteres. Su configuración está fijada para reproducir los resultados; no se reentrena aquí."
    : "Modelo sencillo: el vocabulario TF-IDF y el clasificador se ajustan de verdad en tu navegador con los textos de entrenamiento. Puedes cambiar la representación y comparar resultados.";
  controls.trainStatus.textContent="Configuración modificada";
  controls.trainDescription.textContent=testWasViewed
    ? "El conjunto de prueba ya se ha consultado. Un nuevo ajuste a partir de él puede sesgar las conclusiones."
    : "Pulsa el botón para evaluar esta configuración en validación.";
}

async function train() {
  if(!corpus)return;
  controls.train.disabled=true;
  controls.trainStatus.textContent="Entrenando el clasificador con publicaciones reales…";
  controls.trainDescription.textContent="El vocabulario y el modelo se calculan en tu navegador. No se envían textos al servidor.";
  await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  try {
    selectedTarget=controls.target.value;
    const advanced=controls.model.value==="joint-ridge";
    if(advanced) {
      model=await loadRidgeModel(selectedTarget);
    } else {
      const vectorizer=buildVectorizer(corpus.train,
        {ngrams:Number(controls.ngrams.value),maxFeatures:Number(controls.vocab.value)});
      model=trainClassifier(corpus.train,selectedTarget,controls.model.value,vectorizer,
        {alpha:Number(controls.alpha.value)});
    }
    validation=evaluateClassifier(model,corpus.val,selectedTarget);
    controls.trainStatus.textContent=advanced
      ? "Modelo conjunto Ridge cargado y evaluado":"Modelo entrenado: "+MODEL_NAMES[model.name];
    controls.trainDescription.textContent=
      `${number.format(corpus.train.length)} textos de entrenamiento y ${number.format(corpus.val.length)} de validación. ${number.format(model.vectorizer.size)} características. F1 macro en validación: ${percent(validation.macroF1)}.` +
      (advanced?" Este Ridge se entrenó previamente con el conjunto de entrenamiento, no en el navegador.":"") +
      (testWasViewed ? " Aviso: la prueba final ya fue consultada; no la utilices para ajustar parámetros." : "");
    showValidation();
    controls.results.scrollIntoView({behavior:"smooth",block:"start"});
  } catch(error) {
    console.error("Error de entrenamiento RUW:",error);
    controls.trainStatus.textContent="No se pudo entrenar el clasificador";
    controls.trainDescription.textContent=error.message;
    controls.results.hidden=true;
    controls.testSection.hidden=true;
  } finally {
    controls.train.disabled=false;
  }
}
function evaluateFinalTest() {
  if(!model || !corpus)return;
  const result=evaluateClassifier(model,corpus.test,selectedTarget);
  testWasViewed=true;
  $("#test-accuracy").textContent=percent(result.accuracy);
  $("#test-macro").textContent=percent(result.macroF1);
  $("#test-correct").textContent=number.format(result.correct);
  $("#test-total").textContent=number.format(result.total);
  classConfusion($("#test-confusion"),result);
  classMetrics($("#test-classes"),result);
  controls.testResults.hidden=false;
  controls.testButton.disabled=true;
  $("#test-warning").textContent=
    `Resultados de prueba para ${TARGETS[selectedTarget]}, usando ${MODEL_NAMES[model.name]}. Consulta cerrada para esta configuración. Estos ${number.format(result.total)} textos no se emplearon para entrenar o validar el clasificador. Cualquier elección posterior basada en esta puntuación introduciría sesgo.`;
}
function predictManual() {
  if(!model) return;
  const text=controls.manual.value.trim();
  if(!text) {
    controls.manualResult.textContent="Escribe una frase antes de clasificar.";
    return;
  }
  const result=model.predictText(text);
  controls.manualResult.textContent=
    `Predicción para ${TARGETS[selectedTarget]}: ${LABEL_NAMES[result.label]}. `+
    `Términos conocidos por el modelo: ${result.knownTerms}. Es una clasificación de este texto, no una conclusión sobre la persona que lo escribió.`;
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
  for(const field of [controls.target,controls.model,controls.ngrams,controls.vocab,controls.alpha]) {
    field.addEventListener("change",modelChanged);
  }
  controls.train.addEventListener("click",train);
  controls.testButton.addEventListener("click",evaluateFinalTest);
  controls.manualButton.addEventListener("click",predictManual);
  controls.errorFilter.addEventListener("change",updateReviews);
  controls.errorPrev.addEventListener("click",()=>{if(reviewPage>1){reviewPage-=1;renderReviews();}});
  controls.errorNext.addEventListener("click",()=>{if(reviewPage*PER_PAGE<reviewRows.length){reviewPage+=1;renderReviews();}});
  modelChanged();
  try{
    corpus=await loadRUW();
    renderCorpus();
    controls.content.hidden=false;
    controls.train.disabled=false;
    controls.status.classList.add("text-status-success");
    controls.loadTitle.textContent="Corpus RUW cargado correctamente";
    controls.loadDescription.textContent=
      "6.000 publicaciones originales, con etiquetas de Rusia y Ucrania. La búsqueda y el entrenamiento funcionan ahora de forma local.";
    controls.trainStatus.textContent="Datos preparados para entrenar";
    controls.trainDescription.textContent="Elige el objetivo y los parámetros del modelo; después pulsa «Entrenar y validar».";
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
