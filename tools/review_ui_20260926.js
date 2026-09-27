(() => {
  const kind = REVIEW_UI_KIND;
  const isNatural = kind === 'natural100_seed2026';
  const qa = new URLSearchParams(location.search).get('qa') === '1';
  const sourceItems = [...document.querySelectorAll(isNatural ? 'article.review-item[data-case]' : 'section.sample[data-case-id]')];
  const fields = isNatural
    ? ['case_id','bundle_id','category','object_id','modality','query','decision','note']
    : ['case_id','bundle_id','object_id','semantic_label','aux_modality','rgb_image_path','aux_image_path','rgb_box_xyxy_normalized','aux_box_xyxy_normalized','decision','note'];
  const choices = isNatural
    ? [['correct_unique','正确且唯一'],['wrong','错误'],['multiple','有多个合理目标'],['uncertain','不确定']]
    : [['accept','接受'],['reject','拒绝'],['uncertain','不确定']];
  const title = isNatural ? '自然描述验收' : '跨模态对应验收';
  const description = isNatural
    ? '看图判断描述是否准确，且只指向一个目标。框是原始提议，仅供核对。'
    : '比较 RGB 和红外／深度图，判断两处框是否指向同一对象、范围是否一致。框是待核提议。';
  const state = {index:0, unfinished:false, storageError:false};
  const key = id => `aic-review-20260926:${kind}:${qa ? 'qa:' : ''}${id}`;
  const q = (selector, root=document) => root.querySelector(selector);
  const make = (tag, cls, text) => {const x=document.createElement(tag); if(cls)x.className=cls; if(text!==undefined)x.textContent=text; return x;};
  const items = sourceItems.map((node, index) => {
    const d=node.dataset;
    const id=isNatural ? d.case : d.caseId;
    const original=isNatural ? q('.query',node).textContent.trim() : d.semanticLabel;
    const select=q(isNatural?'select':'.decision',node);
    const sourceNote=q(isNatural?'textarea':'.note',node);
    const item={node,index,id,original,select,sourceNote};
    const prompt=make('div','review-prompt');
    const hint=make('p','review-hint',description);
    const translated=REVIEW_UI_TRANSLATIONS[original];
    if(translated && translated.trim() && translated.trim()!==original){prompt.textContent=translated.trim();}
    else{prompt.textContent=original;prompt.classList.add('missing');hint.textContent='暂无中文译文，请按下方英文原文判断。' ;hint.classList.add('missing');}
    const details=make('details','review-english');
    const summary=make('summary',null,'英文原文');
    const originalP=make('p',null,original);
    details.append(summary,originalP);
    const heading=q('h2',node);
    if(!isNatural){
      const shortId=d.bundleId.replace(/^rgbdt500_/, '');
      const modality=d.auxModality==='infrared'?'红外':'深度';
      heading.textContent=`第 ${String(index+1).padStart(3,'0')} 条 · ${shortId} · ${modality}`;
      heading.title=id;
      heading.after(make('div','review-prompt-label','目标描述'),prompt,hint,details);
      node.querySelectorAll('figcaption').forEach(caption=>{
        const source=caption.textContent;
        caption.title=source;
        caption.textContent=source.replaceAll('Infrared','红外').replaceAll('Depth preview','深度预览').replaceAll('Depth','深度');
      });
    }else heading.after(prompt,hint,details);
    const controls=make('div','review-controls');
    controls.append(make('p','question',isNatural?'这条描述能唯一指向目标吗？':'两张图里的对象和框对应吗？'));
    const buttons=make('div','choices');
    item.buttons=[];
    choices.forEach(([value,label],i)=>{
      const button=make('button',null,`${label}  ${i+1}`);
      button.type='button';button.dataset.value=value;
      button.addEventListener('click',()=>setDecision(item,value));
      item.buttons.push(button);buttons.append(button);
    });
    const clear=make('button','clear','清除判定');clear.type='button';clear.addEventListener('click',()=>setDecision(item,''));buttons.append(clear);
    controls.append(buttons,make('label','note-label','备注（可选）'));
    const note=make('textarea');note.placeholder='补充你看到的问题或疑问';note.value=sourceNote.value;
    note.addEventListener('input',()=>{sourceNote.value=note.value;if(persist(item))renderStatus('草稿已存到此浏览器');});
    item.note=note;
    controls.append(note,make('div','shortcut',`快捷键：1–${choices.length} 判定；← → 切换；在备注中输入时快捷键暂停。`));
    q('.views',node).after(controls);
    for(const figure of node.querySelectorAll('figure')){
      const anchor=q('a',figure);if(anchor)anchor.addEventListener('click',event=>{event.preventDefault();openImage(anchor,figure);});
    }
    restore(item);
    return item;
  });
  const oldHeader=q('body > header');if(oldHeader)oldHeader.hidden=true;
  const oldNav=q('body > nav');if(oldNav)oldNav.hidden=true;
  const app=make('div');app.id='reviewAppHeader';
  const top=make('div','topline');
  const back=make('a','back','← 返回验收入口');back.href='../../../same_day_multimodal_pilot_20260926/index.html';
  const h=make('h1',null,title+(qa?' · 测试模式':''));
  const progress=make('span','progress');
  top.append(back,h,progress);
  const bar=make('div','bar');const barFill=make('span');bar.append(barFill);
  const toolbar=make('div','toolbar');
  const prev=make('button',null,'← 上一条');prev.type='button';prev.onclick=()=>move(-1);
  const next=make('button',null,'下一条 →');next.type='button';next.onclick=()=>move(1);
  const number=make('input');number.type='number';number.min='1';number.max=String(items.length);number.setAttribute('aria-label','跳转到第几条');
  const jump=make('button',null,'跳转');jump.type='button';jump.onclick=()=>{state.unfinished=false;filterCheck.checked=false;state.index=Math.max(0,Math.min(items.length-1,Number(number.value)-1));render();};
  number.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();jump.click();}});
  const jumpLabel=make('label',null,'第');jumpLabel.append(number,document.createTextNode('条'));
  const filterCheck=make('input');filterCheck.type='checkbox';filterCheck.onchange=()=>{state.unfinished=filterCheck.checked;render();};
  const filterLabel=make('label',null,'只看未完成');filterLabel.prepend(filterCheck);
  const spacer=make('span','spacer');
  const importButton=make('button',null,'导入 CSV');importButton.type='button';
  const importInput=make('input');importInput.type='file';importInput.accept='.csv,text/csv';importInput.hidden=true;
  importButton.onclick=()=>importInput.click();importInput.onchange=()=>{if(importInput.files[0])importCsv(importInput.files[0]);importInput.value='';};
  const exportButton=make('button','primary','导出 CSV');exportButton.type='button';exportButton.onclick=exportCsv;
  toolbar.append(prev,next,jumpLabel,jump,filterLabel,spacer,importButton,exportButton,importInput);
  const status=make('div','statusline');status.id='reviewSaveStatus';
  app.append(top,bar,toolbar,status);document.body.prepend(app);
  const lightbox=make('div','lightbox');lightbox.hidden=true;
  const inner=make('div','inner');const lightHead=make('div','head');const lightTitle=make('span');const full=make('a',null,'在新标签打开原图');full.target='_blank';full.rel='noopener';const close=make('button',null,'关闭 ×');close.type='button';close.onclick=()=>{lightbox.hidden=true;};
  lightHead.append(lightTitle,full,close);const big=make('img');inner.append(lightHead,big);lightbox.append(inner);document.body.append(lightbox);
  lightbox.addEventListener('click',e=>{if(e.target===lightbox)lightbox.hidden=true;});
  function openImage(anchor,figure){big.src=anchor.href;big.alt=q('figcaption',figure)?.textContent||'图片';lightTitle.textContent=big.alt;full.href=anchor.href;lightbox.hidden=false;close.focus();}
  function restore(item){
    try{const raw=localStorage.getItem(key(item.id));if(!raw)return;const value=JSON.parse(raw);if(!choices.some(x=>x[0]===value.decision)&&value.decision!=='')throw Error('判定值无效');item.select.value=value.decision||'';item.sourceNote.value=value.note||'';item.note.value=item.sourceNote.value;}
    catch(error){state.storageError=true;}
  }
  function persist(item){
    try{const value={decision:item.select.value,note:item.note.value};if(!value.decision&&!value.note)localStorage.removeItem(key(item.id));else localStorage.setItem(key(item.id),JSON.stringify(value));state.storageError=false;return true;}
    catch(error){state.storageError=true;renderStatus('浏览器草稿保存失败，请立即导出 CSV',true);return false;}
  }
  function renderStatus(message,failed=false){status.textContent=(qa?'测试模式：独立草稿。':'')+message;status.classList.toggle('error',failed);}
  function visible(){return state.unfinished?items.filter(x=>!x.select.value):items;}
  function render(){
    const list=visible();
    let item=items[state.index];
    if(!list.includes(item))item=list.find(x=>x.index>=state.index)||list[0];
    if(item)state.index=item.index;
    items.forEach(x=>x.node.classList.toggle('current',x===item));
    const done=items.filter(x=>x.select.value).length;
    progress.textContent=`已判定 ${done} / ${items.length}`;
    barFill.style.width=`${100*done/items.length}%`;
    number.value=item?String(state.index+1):'';
    const pos=list.indexOf(item);
    prev.disabled=pos<=0;next.disabled=pos<0||pos>=list.length-1;
    if(!item&&state.unfinished)renderStatus('全部项目已判定。可取消“只看未完成”回看。');
    else if(state.storageError)renderStatus('草稿读取或保存异常，请导出 CSV 留存',true);
    else if(!status.textContent||status.textContent.includes('全部项目'))renderStatus(qa?'本次操作保存在独立测试草稿。':'判定与备注自动保存在此浏览器；完成后请导出 CSV。');
    if(item){item.buttons.forEach(b=>b.classList.toggle('selected',b.dataset.value===item.select.value));}
  }
  function move(delta){const list=visible();const pos=list.findIndex(x=>x.index===state.index);const target=list[pos+delta];if(target){state.index=target.index;render();scrollTo({top:0,behavior:'auto'});}}
  function setDecision(item,value){item.select.value=value;persist(item);render();if(!state.storageError)renderStatus(value?'判定已存到此浏览器':'已清除判定');}
  function dataRows(){return items.map(item=>{
    if(isNatural){const record=records[item.index];return [record.case_id,record.bundle_id,record.category,record.object_id,record.modality,record.query,item.select.value,item.note.value];}
    const d=item.node.dataset;return [d.caseId,d.bundleId,d.objectId,d.semanticLabel,d.auxModality,d.rgbImagePath,d.auxImagePath,d.rgbBox,d.auxBox,item.select.value,item.note.value];
  });}
  const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';
  function exportCsv(){
    const csv='\uFEFF'+[fields,...dataRows()].map(row=>row.map(quote).join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const link=make('a');link.href=url;link.download=`${kind}_human_decisions_${qa?'qa_':''}${new Date().toISOString().slice(0,10)}.csv`;
    document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);
    renderStatus('已发起 CSV 下载，请确认文件出现在下载目录。');
  }
  function parseCsv(text){
    const rows=[];let row=[],cell='',inside=false;
    text=text.replace(/^\uFEFF/,'');
    for(let i=0;i<text.length;i++){
      const c=text[i];
      if(c==='"'){if(inside&&text[i+1]==='"'){cell+='"';i++;}else inside=!inside;}
      else if(c===','&&!inside){row.push(cell);cell='';}
      else if((c==='\n'||c==='\r')&&!inside){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);rows.push(row);row=[];cell='';}
      else cell+=c;
    }
    if(inside)throw Error('CSV 引号未闭合');
    if(cell||row.length){row.push(cell);rows.push(row);}
    return rows;
  }
  async function importCsv(file){
    try{
      const rows=parseCsv(await file.text());
      if(rows.length!==items.length+1||rows[0].join('\u001f')!==fields.join('\u001f'))throw Error('表头或行数与当前包不一致');
      const byId=new Map(items.map(x=>[x.id,x]));const seen=new Set();
      for(const row of rows.slice(1)){
        if(row.length!==fields.length||!byId.has(row[0])||seen.has(row[0]))throw Error('case ID 不完整、重复或不属于当前包');
        if(row.at(-2)&&!choices.some(x=>x[0]===row.at(-2)))throw Error('CSV 中有无效判定值');
        seen.add(row[0]);
      }
      for(const row of rows.slice(1)){
        const item=byId.get(row[0]);item.select.value=row.at(-2);item.note.value=row.at(-1);item.sourceNote.value=row.at(-1);if(!persist(item))throw Error('浏览器草稿写入失败');
      }
      render();renderStatus(`已导入 ${items.length} 条，其中 ${items.filter(x=>x.select.value).length} 条已有判定。`);
    }catch(error){renderStatus(`导入失败：${error.message}`,true);}
  }
  addEventListener('keydown',e=>{
    if(!lightbox.hidden){if(e.key==='Escape'){lightbox.hidden=true;e.preventDefault();}return;}
    if(e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,textarea,select,button,a,[contenteditable]'))return;
    const item=items[state.index];if(!item)return;
    if(e.key==='ArrowLeft'){move(-1);e.preventDefault();}
    else if(e.key==='ArrowRight'){move(1);e.preventDefault();}
    else if(/^[1-4]$/.test(e.key)&&choices[Number(e.key)-1]){setDecision(item,choices[Number(e.key)-1][0]);if(!state.unfinished)move(1);e.preventDefault();}
  });
  render();
})();
