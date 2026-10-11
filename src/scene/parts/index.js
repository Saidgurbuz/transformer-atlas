// 3D explorer plug-ins, installed in this order. Each default-exports
// {id, title, install(api) -> dispose}. See docs/SCENE_GUIDE.md for the api.
import embedFlow from './embedFlow.js';
import attentionBeams from './attentionBeams.js';
import mlpNeurons from './mlpNeurons.js';
import residualFlow from './residualFlow.js';
import generationLoop from './generationLoop.js';
import postfx from './postfx.js';
import cameraTour from './cameraTour.js';

export const PARTS = [embedFlow, attentionBeams, mlpNeurons, residualFlow, generationLoop, postfx, cameraTour];
