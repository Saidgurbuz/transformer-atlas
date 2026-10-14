(async () => {
  const { atlas, store } = window.__te;
  const events=[];
  const listen=e=>events.push(e.detail);
  window.addEventListener('atlas:view',listen);
  let checks=0;
  const assert=(ok,message)=>{if(!ok)throw new Error(message);checks++;};
  try {
    assert(!document.getElementById('atlas-google-tag'),'local and file checks do not send visits');
    await atlas.go('embedding');
    assert(events.length===1 && events[0].id==='embedding','one completed scene event');
    await atlas.go('embedding');
    assert(events.length===1,'repeated navigation is not a new visit');
    store.setText('the cat');
    assert(events.length===1,'input edits do not produce scene visits');
    assert(Object.keys(events[0]).sort().join(',')==='id,title','scene payload contains no input or tensor data');
    await atlas.go('stepper');
    assert(events.length===2 && events[1].id==='stepper','code walkthrough visit');
    assert(events[1].title===atlas.stack().at(-1).info.title,'scene title matches the displayed view');
    await atlas.go('model');
    assert(events.length===3 && events[2].id==='model','overview return visit');
    return {checks};
  } finally {window.removeEventListener('atlas:view',listen);}
})()
