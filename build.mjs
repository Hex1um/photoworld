import fs from 'fs';

const tpl = fs.readFileSync(new URL('./app.template.html', import.meta.url), 'utf8');
const weights = fs.readFileSync(new URL('./train/weights.json', import.meta.url), 'utf8');
const examples = fs.readFileSync(new URL('./examples.json', import.meta.url), 'utf8');

const modelJs = 'var MODEL = ' + weights + ';';
const exJs = 'var EXAMPLES = ' + examples + ';';

const out = tpl
  .replace('/*__MODEL__*/', modelJs)
  .replace('/*__EXAMPLES__*/', exJs);

fs.writeFileSync(new URL('./index.html', import.meta.url), out);
const kb = (Buffer.byteLength(out) / 1024).toFixed(0);
console.log('built index.html', kb + 'KB');
