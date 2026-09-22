// Offline execution of the sibling encoder's pure functions with fixture chars.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../../keytao-next');
const ts = require(path.join(root, 'node_modules/typescript'));
const forbidden = () => { throw new Error('S69: network/cache access forbidden'); };
function load(relative) {
  const file = path.join(root, relative);
  const source = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const exports = {};
  vm.runInNewContext(source, {
    exports, console, process: { cwd: () => root },
    require: (name) => {
      if (name === 'path') return path;
      if (name === 'fs') return { readFileSync: (file, encoding) => {
        assert.ok(['keytao-root.csv', 'keytao-split.csv'].some(name => file === path.join(root, 'config', name)));
        return fs.readFileSync(file, encoding);
      }};
      if (name === 'https') return { get: forbidden, request: forbidden };
      if (name === 'pinyin-pro') return { customPinyin: () => {}, pinyin: forbidden };
      if (name === './zdicLookupCache') return { readZdicPinyinCache: forbidden, writeZdicPinyinCache: forbidden };
      if (name === '@/lib/constants/codeValidation') return load('lib/constants/codeValidation.ts');
      throw new Error('Unexpected import: ' + name);
    },
  }, { filename: file });
  return exports;
}
const encoder = load('lib/services/keytaoEncoder.ts');
const results = ['shé', 'zhé'].map(reading => {
  const chars = [
    { char: '折', pinyin: reading, phoneticCode: reading === 'shé' ? 'ee' : 'qe', shapeCode: 'iu' },
    { char: '煞', pinyin: 'shā', phoneticCode: 'es', shapeCode: 'uu' },
  ];
  const encoded = encoder.buildPhraseEncodingFromChars('折煞', chars);
  return { reading, codes: encoded.codes, altCodes: encoded.altCodes,
    analysis: encoder.analyzeRequestedCode(encoded, 'fees') };
});
assert.equal(results[0].analysis.matchType, 'unsupported');
assert.equal(results[1].analysis.matchType, 'flyKey');
assert.ok(results[1].altCodes.includes('fees'));
console.log(JSON.stringify({ mode: 'pure-encoder-fixture', paidModelCalls: 0, results }, null, 2));
