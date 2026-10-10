/**
 * tina4-harness-lab — right-click a notebook cell to use the notebook's OWN local model.
 *
 *   • Check Code   — review the active cell (rendered as Markdown)
 *   • Create Cell  — describe a cell; the model writes it; you Insert / run it
 *   • Fix Code     — rewrite the HIGHLIGHTED selection (or the whole cell); you Replace / Insert
 *
 * Suggestions are shown first (syntax-highlighted) with Replace / Insert below / Cancel — nothing
 * changes until you choose. The model runs in the notebook's kernel (loaded on first use if needed),
 * so a clean notebook + a local LLM is all you need.
 */
import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { Dialog, ICommandPalette, Notification, showDialog } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { LabIcon } from '@jupyterlab/ui-components';
import { Menu, Widget } from '@lumino/widgets';

const CMD = {
  check: 'tina4-harness:check-code',
  create: 'tina4-harness:create-cell',
  fix: 'tina4-harness:fix-code',
  generate: 'tina4-harness:generate-here',
  preview: 'tina4-harness:app-preview',
  chat: 'tina4-harness:open-chat'
};

const chatIcon = new LabIcon({
  name: 'tina4-harness:chat',
  svgstr:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">' +
    '<path class="jp-icon3" fill="#616161" d="M4 3h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-5 4v-4H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm3 6h10v-2H7v2zm0 4h7v-2H7v2z"/>' +
    '</svg>'
});

/** Run `__harness_edit(kind, payload)` in the kernel and return the model's text. */
async function kernelEdit(
  panel: NotebookPanel,
  kind: string,
  payload: Record<string, unknown>,
  onStatus?: (s: string) => void,
  onStream?: (delta: string) => void
): Promise<string> {
  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel) {
    throw new Error('No kernel is running for this notebook.');
  }
  const b64 = btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
  // Make `harness` importable in ANY fresh kernel, without depending on the notebook's setup cell
  // having run: walk up from the kernel's cwd and add every `<dir>/src` that contains `harness`.
  const bootstrap =
    'import sys, os\n' +
    '_d = os.getcwd()\n' +
    'for _ in range(8):\n' +
    "    _s = os.path.join(_d, 'src')\n" +
    "    if os.path.isdir(os.path.join(_s, 'harness')) and _s not in sys.path:\n" +
    '        sys.path.insert(0, _s)\n' +
    '    _p = os.path.dirname(_d)\n' +
    '    if _p == _d:\n' +
    '        break\n' +
    '    _d = _p\n';
  const code =
    bootstrap +
    'import harness.lab as _hl\n' +
    `print(_hl.harness_edit(${JSON.stringify(kind)}, "${b64}"))`;
  const future = kernel.requestExecute({ code, stop_on_error: false, store_history: false });
  let buf = '';
  future.onIOPub = (msg: any): void => {
    const t = msg.header.msg_type;
    if (t === 'stream' && msg.content.name === 'stdout') {
      const text = msg.content.text as string;
      for (const ln of text.split('\n')) {
        if (ln.startsWith('__HSTATUS__')) {
          if (onStatus) {
            onStatus(ln.slice('__HSTATUS__'.length).trim());
          }
        } else if (ln.startsWith('__HSTREAM__') && onStream) {
          try {
            onStream(decodeURIComponent(escape(window.atob(ln.slice('__HSTREAM__'.length).trim()))));
          } catch {
            /* ignore a partial/garbled chunk */
          }
        }
      }
      buf += text;
    } else if (t === 'error') {
      buf += (msg.content.evalue as string) || 'kernel error';
    }
  };
  await future.done;
  const marker = '__HARNESS__';
  const i = buf.lastIndexOf(marker);
  if (i < 0) {
    throw new Error('No reply from the kernel. Is the harness on the path and the kernel running?');
  }
  const line = buf.slice(i + marker.length).split('\n')[0].trim();
  const res = JSON.parse(line);
  if (res.error) {
    throw new Error(res.error as string);
  }
  return res.text as string;
}

/** The live selection in a cell's CodeMirror editor, or the whole cell if nothing is highlighted. */
function selectionOf(cell: Cell): { from: number; to: number; scope: string } {
  const src = cell.model.sharedModel.getSource();
  const editor: any = cell.editor;
  // 1) CodeMirror 6 EditorView — the ground truth for what is highlighted right now.
  try {
    const view = editor?.editor;
    const main = view?.state?.selection?.main;
    if (main && main.to > main.from) {
      return { from: main.from, to: main.to, scope: 'selection' };
    }
  } catch {
    /* fall through */
  }
  // 2) CodeEditor API fallback.
  try {
    const sel = editor.getSelection();
    const f = editor.getOffsetAt(sel.start);
    const t = editor.getOffsetAt(sel.end);
    if (t > f) {
      return { from: f, to: t, scope: 'selection' };
    }
  } catch {
    /* fall through */
  }
  return { from: 0, to: src.length, scope: 'cell' };
}

/** A dialog body with a multi-line textarea. */
class TextareaBody extends Widget {
  private _ta: HTMLTextAreaElement;
  constructor(placeholder: string, initial = '') {
    super();
    this.addClass('harness-dialog');
    this._ta = document.createElement('textarea');
    this._ta.placeholder = placeholder;
    this._ta.value = initial;
    this._ta.rows = 4;
    this.node.appendChild(this._ta);
    setTimeout(() => {
      this._ta.focus();
      this._ta.select();
    }, 0);
  }
  getValue(): string {
    return this._ta.value;
  }
}

async function promptText(title: string, placeholder: string, okLabel = 'Run', initial = ''): Promise<string | null> {
  const body = new TextareaBody(placeholder, initial);
  const result = await showDialog({
    title,
    body,
    buttons: [Dialog.cancelButton(), Dialog.okButton({ label: okLabel })]
  });
  if (!result.button.accept) {
    return null;
  }
  return body.getValue().trim() || null;
}

/** A dialog body with an optional rendered review and an EDITABLE code area. */
class EditCodeBody extends Widget {
  private _ta: HTMLTextAreaElement;
  constructor(code: string, reviewNode?: HTMLElement) {
    super();
    this.addClass('harness-dialog');
    if (reviewNode) {
      this.node.appendChild(reviewNode);
    }
    this._ta = document.createElement('textarea');
    this._ta.className = 'harness-codearea';
    this._ta.value = code;
    this._ta.spellcheck = false;
    this._ta.rows = Math.min(26, Math.max(4, code.split('\n').length + 1));
    this.node.appendChild(this._ta);
    setTimeout(() => this._ta.focus(), 0);
  }
  getValue(): string {
    return this._ta.value;
  }
}

/** Show an editable code area (with an optional review) — returns the edited code, or null. */
async function editCodeDialog(
  rendermime: IRenderMimeRegistry,
  title: string,
  code: string,
  reviewMd?: string
): Promise<string | null> {
  let reviewNode: HTMLElement | undefined;
  if (reviewMd) {
    const r = rendermime.createRenderer('text/markdown');
    await r.renderModel(rendermime.createModel({ data: { 'text/markdown': reviewMd }, trusted: true }));
    reviewNode = document.createElement('div');
    reviewNode.className = 'harness-md';
    reviewNode.appendChild(r.node);
  }
  const body = new EditCodeBody(code, reviewNode);
  const result = await showDialog({
    title,
    body,
    buttons: [Dialog.cancelButton(), Dialog.okButton({ label: 'Apply to cell' })]
  });
  return result.button.accept ? body.getValue() : null;
}

/** Render Markdown (incl. tables and ```code``` with highlighting) into a dialog body widget. */
async function markdownWidget(rendermime: IRenderMimeRegistry, md: string): Promise<Widget> {
  const renderer = rendermime.createRenderer('text/markdown');
  await renderer.renderModel(
    rendermime.createModel({ data: { 'text/markdown': md }, trusted: true })
  );
  const node = document.createElement('div');
  node.className = 'harness-md';
  node.appendChild(renderer.node);
  const widget = new Widget({ node });
  widget.addClass('harness-dialog');
  return widget;
}

type CodeChoice = 'replace' | 'insert' | 'cancel';

/** Show suggested code (highlighted) and let the user choose what to do with it. */
async function proposeCode(
  rendermime: IRenderMimeRegistry,
  title: string,
  code: string,
  allowReplace: boolean
): Promise<CodeChoice> {
  const body = await markdownWidget(rendermime, '```python\n' + code + '\n```');
  const buttons = [Dialog.cancelButton({ label: 'Cancel' }), Dialog.createButton({ label: 'Insert below' })];
  if (allowReplace) {
    buttons.push(Dialog.createButton({ label: 'Replace' }));
  }
  const result = await showDialog({ title, body, buttons });
  const label = result.button.label;
  if (label === 'Replace') {
    return 'replace';
  }
  if (label === 'Insert below') {
    return 'insert';
  }
  return 'cancel';
}

/** Open a running web app (e.g. a `tina4 serve` server) as an IFRAME tab inside JupyterLab's main
 *  area, with a small toolbar (reload + open-in-browser). Keeps the preview beside the notebook. */
function openAppPreview(app: JupyterFrontEnd, url: string): void {
  const w = new Widget();
  w.id = 'harness-preview-' + Date.now();
  w.title.label = 'App Preview';
  w.title.closable = true;
  w.node.style.display = 'flex';
  w.node.style.flexDirection = 'column';
  w.node.style.height = '100%';

  const bar = document.createElement('div');
  bar.style.cssText =
    'display:flex;gap:6px;align-items:center;padding:4px 8px;border-bottom:1px solid var(--jp-border-color2);';
  const link = document.createElement('span');
  link.textContent = url;
  link.style.cssText = 'flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:var(--jp-ui-font-size0);color:var(--jp-ui-font-color2);';
  const reload = document.createElement('button');
  reload.textContent = '⟳ Reload';
  reload.className = 'jp-mod-styled';
  const openExt = document.createElement('button');
  openExt.textContent = '↗ Browser';
  openExt.className = 'jp-mod-styled';
  bar.appendChild(link);
  bar.appendChild(reload);
  bar.appendChild(openExt);

  const iframe = document.createElement('iframe');
  iframe.src = url;
  iframe.style.cssText = 'flex:1 1 auto;width:100%;border:none;background:#fff;';
  reload.onclick = () => {
    iframe.src = url; // reassign to reload
  };
  openExt.onclick = () => window.open(url, '_blank');

  w.node.appendChild(bar);
  w.node.appendChild(iframe);
  app.shell.add(w, 'main', { mode: 'split-right' });
  app.shell.activateById(w.id);
}

/** Reload every open App Preview iframe (after the harness writes to the project). */
function reloadPreviews(app: JupyterFrontEnd): number {
  let n = 0;
  try {
    for (const w of (app.shell as any).widgets('main')) {
      if (w.id && String(w.id).startsWith('harness-preview-')) {
        const iframe = w.node.querySelector('iframe') as HTMLIFrameElement | null;
        if (iframe) {
          // eslint-disable-next-line no-self-assign
          iframe.src = iframe.src; // force a reload
          n++;
        }
      }
    }
  } catch {
    /* ignore */
  }
  return n;
}

/** URLs of open App Preview tabs — a signal for which project is being worked on. */
function openPreviewUrls(app: JupyterFrontEnd): string[] {
  const urls: string[] = [];
  try {
    for (const w of (app.shell as any).widgets('main')) {
      if (w.id && String(w.id).startsWith('harness-preview-')) {
        const iframe = w.node.querySelector('iframe') as HTMLIFrameElement | null;
        if (iframe && iframe.src) {
          urls.push(iframe.src);
        }
      }
    }
  } catch {
    /* ignore */
  }
  return urls;
}

/** Does this notebook have a live kernel we can run the bridge on? */
function hasKernel(panel: NotebookPanel): boolean {
  return !!(panel && !panel.isDisposed && panel.sessionContext?.session?.kernel);
}

function insertBelowWithSource(panel: NotebookPanel, source: string): Cell {
  NotebookActions.insertBelow(panel.content);
  const cell = panel.content.activeCell!;
  cell.model.sharedModel.setSource(source);
  return cell;
}

/** Append a new code cell at the end of the notebook. */
function appendCell(panel: NotebookPanel, source: string): Cell {
  const nb = panel.content;
  nb.activeCellIndex = nb.widgets.length - 1;
  NotebookActions.insertBelow(nb);
  const cell = nb.activeCell!;
  cell.model.sharedModel.setSource(source);
  return cell;
}

/** Append a new code cell at the end of the notebook and run it. */
async function appendAndRun(panel: NotebookPanel, source: string): Promise<void> {
  appendCell(panel, source);
  await NotebookActions.run(panel.content, panel.sessionContext);
}

/** Select a specific cell and run it. */
async function runCell(panel: NotebookPanel, cell: Cell): Promise<void> {
  const i = panel.content.widgets.indexOf(cell);
  if (i >= 0) {
    panel.content.activeCellIndex = i;
    await NotebookActions.run(panel.content, panel.sessionContext);
  }
}

/** Remove a cell by widget. */
function deleteCellWidget(panel: NotebookPanel, cell: Cell): void {
  const i = panel.content.widgets.indexOf(cell);
  const model: any = panel.content.model;
  if (i >= 0 && model?.sharedModel) {
    model.sharedModel.deleteCell(i);
  }
}

/** Create a brand-new notebook, write the code into its first cell, and run it. */
async function newNotebookWithCode(app: JupyterFrontEnd, code: string): Promise<void> {
  const created: any = await app.commands.execute('docmanager:new-untitled', { type: 'notebook' });
  const widget: any = await app.commands.execute('docmanager:open', { path: created.path });
  await widget.context.ready;
  try {
    await widget.sessionContext.ready;
  } catch {
    /* kernel may still be starting */
  }
  const nb = widget.content;
  nb.activeCellIndex = 0;
  if (nb.activeCell) {
    nb.activeCell.model.sharedModel.setSource(code);
  }
  try {
    await NotebookActions.run(nb, widget.sessionContext);
  } catch {
    /* leave it unrun if the kernel isn't ready */
  }
}

/** The active cell's output/error as text (so the model can help fix problems). */
function cellOutputText(cell: Cell): string {
  const model: any = cell.model;
  const outs = model.outputs;
  if (!outs || typeof outs.toJSON !== 'function') {
    return '';
  }
  let txt = '';
  for (const o of outs.toJSON()) {
    if (o.output_type === 'stream') {
      txt += o.text;
    } else if (o.output_type === 'execute_result' || o.output_type === 'display_data') {
      txt += (o.data && o.data['text/plain']) || '';
    } else if (o.output_type === 'error') {
      txt += `${o.ename}: ${o.evalue}\n` + (o.traceback || []).join('\n');
    }
  }
  // strip ANSI colour codes and cap length
  return txt.replace(/\u001b\[[0-9;]*m/g, '').trim().slice(-1800);
}

/** Top-level names (functions/classes) a snippet defines. */
function definedNames(code: string): Set<string> {
  const names = new Set<string>();
  const re = /^(?:def|class)\s+([A-Za-z_]\w*)/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(code))) {
    names.add(m[1]);
  }
  return names;
}

/** The existing code cell that defines the same names as `code` (so a fix edits it in place). */
function findRevisionTarget(panel: NotebookPanel, code: string): Cell | null {
  const codeNames = definedNames(code);
  if (!codeNames.size) {
    return null;
  }
  const cells = panel.content.widgets;
  for (let i = cells.length - 1; i >= 0; i--) {
    const c = cells[i];
    if (c.model.type !== 'code') {
      continue;
    }
    const cellNames = definedNames(c.model.sharedModel.getSource());
    for (const n of codeNames) {
      if (cellNames.has(n)) {
        return c;
      }
    }
  }
  return null;
}

/** The notebook's existing code (so a reviewer/model knows what's already defined). */
function collectNotebookCode(panel: NotebookPanel): string {
  const out: string[] = [];
  for (const c of panel.content.widgets) {
    if (c.model.type === 'code') {
      const s = c.model.sharedModel.getSource().trim();
      if (s) {
        out.push(s);
      }
    }
  }
  const code = out.join('\n\n');
  return code.length > 4000 ? code.slice(-4000) : code;
}

/** Put code into the blank active cell if there is one, else append a new cell. */
function placeCode(panel: NotebookPanel, source: string): { cell: Cell; created: boolean } {
  const active = panel.content.activeCell;
  if (active && active.model.type === 'code' && active.model.sharedModel.getSource().trim() === '') {
    active.model.sharedModel.setSource(source);
    return { cell: active, created: false };
  }
  return { cell: appendCell(panel, source), created: true };
}

/** Remove any stray ``` fence lines so an inserted cell is pure code. */
function stripFences(code: string): string {
  return code
    .split('\n')
    .filter(line => !line.trim().startsWith('```'))
    .join('\n')
    .trim();
}

/** Split a Markdown reply into prose and its fenced code blocks (code has no fences). */
// Reply prefix (from harness.lab) that means: insert the code block straight into a new cell.
const INSERT_SENTINEL = '@@HARNESS_INSERT@@\n';
// Reply prefix meaning: insert into a NEW cell AND run it (e.g. an example that calls existing code).
const RUN_SENTINEL = '@@HARNESS_RUN@@';
// Reply prefix meaning: files were written to a Tina4 project — show read-only + refresh preview.
const WROTE_SENTINEL = '@@HARNESS_WROTE@@';
// Reply prefix meaning: a new app was provisioned+served — open the App Preview tab at <url>.
const APP_SENTINEL = '@@HARNESS_APP@@';
// Reply prefix meaning: the kernel VALIDATED a real fix — show the prose, offer to apply the trailing
// block to the cell. Only this sentinel offers a fix; a plain explanation that echoes code never does.
const FIX_SENTINEL = '@@HARNESS_FIX@@\n';
// Reply prefix meaning: a clarifying question with options — render them as clickable pills. Payload
// is JSON {lead, options} on the rest of the line.
const CHOICES_SENTINEL = '@@HARNESS_CHOICES@@';

/** Drop prose that is only a heading/label with no real content (e.g. "Explanation of the fix:",
 *  "Output:") — those render as big blank headers next to a code cell and say nothing. */
function meaningfulProse(prose: string): string {
  // Drop prose lines that are just a heading/label with no content (ending in ':', or a known
  // filler), keeping lines that actually say something. CODE INSIDE ``` FENCES IS NEVER TOUCHED —
  // a line like `def f():`, `if x:`, or a docstring `Args:` must survive verbatim.
  const kept: string[] = [];
  let inFence = false;
  for (const raw of prose.split('\n')) {
    if (/^\s*```/.test(raw)) {
      inFence = !inFence;
      kept.push(raw);
      continue;
    }
    if (inFence) {
      kept.push(raw); // preserve code exactly (indentation and all)
      continue;
    }
    const line = raw.trim();
    if (!line) {
      continue;
    }
    const plain = line
      .replace(/^#{1,6}\s*/, '')
      .replace(/\*\*/g, '')
      .replace(/^[-*]\s+/, '')
      .trim();
    const isBareLabel =
      /[:：]$/.test(plain) ||
      /^(short(est)? form(\s+with\s+.+)?|usage|example|examples|output|result|explanation(\s+of\s+.+)?|code|solution|note|notes|summary|steps?|here(\s*'?s)?(\s+the)?(\s+code|\s+fix)?)\s*[:：]?$/i.test(
        plain
      );
    if (isBareLabel || plain.length < 3) {
      continue;
    }
    kept.push(line);
  }
  return kept.join('\n').trim();
}

/** A short, human description of what a code block does, for the action note. */
function describeCode(code: string): string {
  const m1 = code.match(/^\s*print\s*\(\s*([A-Za-z_]\w*)\s*\(/m);
  if (m1) {
    return `called ${m1[1]}()`;
  }
  const m2 = code.match(/^\s*([A-Za-z_][\w.]*)\s*\(/m);
  if (m2 && !/^(print|len|range|list|dict|set|str|int|float|open)$/.test(m2[1])) {
    return `called ${m2[1]}()`;
  }
  const defs = [...code.matchAll(/^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)/gm)];
  if (defs.length === 1) {
    const name = defs[0][1];
    const params = defs[0][2]
      .split(',')
      .map(s => s.split(':')[0].split('=')[0].trim())
      .filter(p => p && p !== 'self')
      .join(', ');
    return `wrote ${name}(${params})`;
  }
  if (defs.length > 1) {
    return `wrote ${defs.map(d => d[1]).join(', ')}`;
  }
  const cls = code.match(/^\s*class\s+([A-Za-z_]\w*)/m);
  if (cls) {
    return `defined class ${cls[1]}`;
  }
  return '';
}

function splitCode(md: string): { prose: string; blocks: string[] } {
  const blocks: string[] = [];
  const prose = md
    .replace(/```[a-zA-Z0-9_+-]*\n?([\s\S]*?)```/g, (_m, code: string) => {
      blocks.push(stripFences(code));
      return '';
    })
    .trim();
  return { prose, blocks };
}

/** A side-panel chat with the notebook's local model — for ideating and drafting. */
class ChatPanel extends Widget {
  private _log: HTMLDivElement;
  private _input: HTMLTextAreaElement;
  private _send: HTMLButtonElement;
  private _stop: HTMLButtonElement;
  private _thinkBtn: HTMLButtonElement;
  private _thinkMode: 'full' | 'brief' | 'off' = 'brief';
  private _ctxLabel: HTMLSpanElement;
  private _lastPanel: NotebookPanel | null = null;
  private _busyPanel: NotebookPanel | null = null;
  private _lastSel: { cell: Cell; from: number; to: number } | null = null;
  private _snapCell: Cell | null = null;     // the cell the user was on when they reached for chat
  private _snapPanel: NotebookPanel | null = null;  // the notebook they were in (with a live kernel)
  private _histories = new Map<string, { role: string; content: string }[]>();
  private _currentKey = '';
  private _messages: { role: string; content: string }[] = [];
  private _welcome =
    'Draft Python code or a `@tool`, ask a Python question, or ask about an open image.';

  constructor(
    private app: JupyterFrontEnd,
    private tracker: INotebookTracker,
    private rendermime: IRenderMimeRegistry
  ) {
    super();
    this.id = 'tina4-harness-chat';
    this.addClass('harness-chat');
    this.title.icon = chatIcon;
    this.title.caption = 'Harness chat (local model)';

    const wrap = document.createElement('div');
    wrap.className = 'harness-chat-wrap';

    const header = document.createElement('div');
    header.className = 'harness-chat-header';
    this._ctxLabel = document.createElement('span');
    this._ctxLabel.className = 'harness-chat-ctx';
    // Thinking toggle: cycles Full → Brief → Off. Controls the model's reasoning phase and
    // caps reply length, so answers stay short instead of rambling.
    this._thinkBtn = document.createElement('button');
    this._thinkBtn.className = 'harness-chat-think';
    this._thinkBtn.title = 'Thinking depth — click to cycle Full / Brief / Off';
    this._thinkBtn.onclick = () => this._cycleThink();
    this._renderThinkBtn();
    const clear = document.createElement('button');
    clear.className = 'harness-chat-clear';
    clear.textContent = 'Clear';
    clear.title = 'Clear this conversation';
    clear.onclick = () => this._clear();
    header.appendChild(this._ctxLabel);
    header.appendChild(this._thinkBtn);
    header.appendChild(clear);

    this._log = document.createElement('div');
    this._log.className = 'harness-chat-log';
    const row = document.createElement('div');
    row.className = 'harness-chat-input';
    this._input = document.createElement('textarea');
    this._input.rows = 2;
    this._input.placeholder = 'Ask or ideate…  (Enter to send, Shift+Enter for a newline)';
    this._send = document.createElement('button');
    this._send.textContent = 'Send';
    this._send.className = 'jp-mod-styled jp-mod-accept';
    // Stop: interrupts the running generation (hidden until one is in flight).
    this._stop = document.createElement('button');
    this._stop.textContent = 'Stop';
    this._stop.className = 'jp-mod-styled jp-mod-reject harness-chat-stop';
    this._stop.title = 'Stop the current generation';
    this._stop.style.display = 'none';
    this._stop.onclick = () => this._stopGeneration();
    row.appendChild(this._input);
    row.appendChild(this._send);
    row.appendChild(this._stop);
    wrap.appendChild(header);
    wrap.appendChild(this._log);
    wrap.appendChild(row);
    this.node.appendChild(wrap);

    this._send.onclick = () => void this._submit();
    this._input.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        void this._submit();
      }
    });
    // Snapshot the cell selection the moment you reach for the chat (before the cell blurs).
    this._input.addEventListener('pointerdown', () => this._snapshotSelection());
    this._input.addEventListener('focus', () => this._snapshotSelection());

    // Conversation follows the active notebook / file.
    const shell: any = this.app.shell;
    if (shell.currentChanged && shell.currentChanged.connect) {
      shell.currentChanged.connect(() => void this._syncContext());
    }
    this.tracker.currentChanged.connect(() => {
      if (this.tracker.currentWidget) {
        this._lastPanel = this.tracker.currentWidget; // remember a notebook for its kernel
      }
      void this._syncContext();
    });

    this._currentKey = '';
    this._histories.set('', []);
    this._messages = this._histories.get('')!;
    this._updateCtxLabel('');
    void this._renderLog();
    void this._syncContext(); // load the active document's saved conversation, if any
  }

  /** Path of the active main-area document (notebook / image / file), or '' for none. */
  private _currentDocPath(): string {
    const w: any = (this.app.shell as any).currentWidget;
    const p = w?.context?.path;
    return typeof p === 'string' ? p : '';
  }

  private async _syncContext(): Promise<void> {
    const path = this._currentDocPath();
    if (!path || path === this._currentKey) {
      return;
    }
    this._currentKey = path;
    if (!this._histories.has(path)) {
      let hist: { role: string; content: string }[] = [];
      const panel = this.tracker.currentWidget;
      if (panel) {
        try {
          hist = await this._loadHistory(panel, path);
        } catch {
          /* no saved history, or no kernel — start fresh */
        }
      }
      this._histories.set(path, hist);
    }
    this._messages = this._histories.get(path)!;
    this._updateCtxLabel(path);
    await this._renderLog();
  }

  /** The notebook's existing code (so the model reuses names instead of redefining them). */
  private _notebookCode(panel: NotebookPanel): string {
    const out: string[] = [];
    for (const c of panel.content.widgets) {
      if (c.model.type === 'code') {
        const s = c.model.sharedModel.getSource().trim();
        if (s) {
          out.push(s);
        }
      }
    }
    const code = out.join('\n\n');
    return code.length > 4000 ? code.slice(-4000) : code;
  }

  /** Content of EVERY file the user has OPEN in a main-area tab (file editors + other notebooks),
   *  so questions are aware of everything on screen — not just the active notebook's cells.
   *  Multiple open files all get included, each given a fair share of the budget. */
  private _openFiles(activePanel: NotebookPanel | null): string {
    const docs: { name: string; body: string }[] = [];
    const seen = new Set<string>();
    const push = (name: string, text: string): void => {
      const body = (text || '').trim();
      if (body && !seen.has(name)) {
        seen.add(name);
        docs.push({ name, body });
      }
    };
    let it: IterableIterator<any> | any[];
    try {
      it = (this.app.shell as any).widgets('main');
    } catch {
      it = [];
    }
    for (const w of it as any) {
      try {
        // Prefer the full document path (e.g. "tina4app/src/routes/home.py") so the kernel can tell
        // when an open file lives inside a Tina4 project and switch paradigms automatically.
        const name: string = w.context?.path || w.title?.label || w.id || 'untitled';
        // A plain file editor (FileEditor): text lives on its editor/model shared model.
        const ed = w.content?.editor || w.content;
        const fem = ed?.model?.sharedModel || w.content?.model?.sharedModel;
        if (fem && typeof fem.getSource === 'function' && !w.content?.widgets) {
          push(name, fem.getSource());
          continue;
        }
        // A notebook widget: concatenate its code cells (skip the active one — sent separately).
        if (w.content?.widgets && w !== activePanel) {
          const cells: string[] = [];
          for (const c of w.content.widgets) {
            if (c.model?.type === 'code') {
              const s = c.model.sharedModel.getSource().trim();
              if (s) {
                cells.push(s);
              }
            }
          }
          push(name, cells.join('\n\n'));
        }
      } catch {
        /* skip a widget we can't read */
      }
    }
    if (!docs.length) {
      return '';
    }
    // Fair share so one big file never crowds the others out (total ~12000 chars).
    const TOTAL = 12000;
    const per = Math.max(1200, Math.floor(TOTAL / docs.length));
    const header = `(${docs.length} file(s) open: ${docs.map(d => d.name).join(', ')})`;
    const parts = docs.map(d => {
      const body = d.body.length > per ? d.body.slice(0, per) + '\n# …(truncated)…' : d.body;
      return `# ===== ${d.name} =====\n${body}`;
    });
    return header + '\n\n' + parts.join('\n\n');
  }

  private async _loadHistory(panel: NotebookPanel, key: string): Promise<{ role: string; content: string }[]> {
    const txt = await kernelEdit(panel, 'chat_load', { key });
    const arr = JSON.parse(txt);
    return Array.isArray(arr) ? arr : [];
  }

  /** Persist the current conversation to .tina4/chat/ (best-effort). */
  private _save(): void {
    const panel = this.tracker.currentWidget;
    if (panel) {
      void kernelEdit(panel, 'chat_save', { key: this._currentKey, messages: this._messages }).catch(() => undefined);
    }
  }

  private _updateCtxLabel(key: string): void {
    if (!key) {
      this._ctxLabel.textContent = 'Harness chat';
      return;
    }
    const name = key.split('/').pop() || key;
    const icon = /\.ipynb$/.test(name) ? '📓 ' : /\.(png|jpe?g|gif|webp|bmp)$/i.test(name) ? '🖼 ' : '📄 ';
    this._ctxLabel.textContent = icon + name;
  }

  private _clear(): void {
    this._histories.set(this._currentKey, []);
    this._messages = this._histories.get(this._currentKey)!;
    void this._renderLog();
    this._save();
  }

  /** Re-render the log from the current conversation (code stripped from history). */
  private async _renderLog(): Promise<void> {
    this._log.innerHTML = '';
    if (!this._messages.length) {
      const hint = document.createElement('div');
      hint.className = 'harness-welcome';
      hint.textContent = this._welcome.replace(/[`*]/g, '');
      this._log.appendChild(hint);
      return;
    }
    for (const m of this._messages) {
      if (m.role === 'user') {
        await this._append('user', m.content);
      } else {
        const { prose, blocks } = splitCode(m.content);
        await this._append('assistant', prose || (blocks.length ? '📝 (code drafted earlier)' : m.content));
      }
    }
  }

  /** Show, as a muted block, the selection and/or error the agent is being given. */
  /** Remember the cell selection at the moment the chat input is reached for. */
  private _snapshotSelection(): void {
    // Capture what the user is looking at BEFORE focus moves to the chat input: the focused
    // notebook, its active cell, and any text selection. These power "what's wrong with my code?"
    const panel = this.tracker.currentWidget;
    if (panel && hasKernel(panel)) {
      this._snapPanel = panel;
    }
    const cell = panel?.content.activeCell;
    if (!cell) {
      return;
    }
    this._snapCell = cell;
    const { from, to, scope } = selectionOf(cell);
    if (scope === 'selection') {
      this._lastSel = { cell, from, to };
    }
  }

  /** If the active main-area document is an image, return its path — else null. */
  private _currentImagePath(): string | null {
    const current: any = (this.app.shell as any).currentWidget;
    const path = current?.context?.path;
    if (typeof path === 'string' && /\.(png|jpe?g|gif|webp|bmp)$/i.test(path)) {
      return path;
    }
    return null;
  }

  /** Put the drafted code straight into a cell (the blank active cell, or a new one) and run it. */
  private async _offerCode(
    panel: NotebookPanel,
    code: string,
    sel: { cell: Cell; from: number; to: number } | null,
    isRevision: boolean
  ): Promise<void> {
    // A fix/edit that redefines an existing cell's names edits THAT cell in place (no duplicate).
    const target = isRevision ? findRevisionTarget(panel, code) : null;
    let mode: 'created' | 'filled' | 'replaced';
    let cell: Cell;
    let prevSource = '';
    if (target) {
      prevSource = target.model.sharedModel.getSource();
      target.model.sharedModel.setSource(code);
      cell = target;
      mode = 'replaced';
    } else {
      const r = placeCode(panel, code);
      cell = r.cell;
      mode = r.created ? 'created' : 'filled';
    }
    // Don't auto-run code that would block on input() AT TOP LEVEL (module scope). input() inside a
    // function body is fine — running the cell just defines the function, so we DO run it.
    const readsInput = code.split('\n').some(l => /\binput\s*\(/.test(l) && !/^\s/.test(l) && !l.trimStart().startsWith('#'));
    if (!readsInput) {
      await runCell(panel, cell);
    }

    let on = true;
    const revert = (): void => {
      if (mode === 'created') {
        deleteCellWidget(panel, cell);
      } else if (mode === 'filled') {
        cell.model.sharedModel.setSource('');
      } else {
        cell.model.sharedModel.setSource(prevSource);
      }
      on = false;
    };
    const reapply = async (): Promise<void> => {
      if (mode === 'created') {
        cell = placeCode(panel, code).cell;
      } else {
        cell.model.sharedModel.setSource(code);
      }
      await runCell(panel, cell);
      on = true;
    };

    const bubble = document.createElement('div');
    bubble.className = 'harness-msg harness-assistant';
    const note = document.createElement('div');
    note.className = 'harness-codenote';
    const cellNo = panel.content.widgets.indexOf(cell) + 1; // 1-based for humans
    const what = describeCode(code);
    const did = mode === 'replaced' ? 'Updated' : mode === 'created' ? 'Added' : 'Filled';
    const ran = readsInput ? ' — run it and type your answer in the prompt.' : ' and ran it.';
    note.textContent =
      `✓ ${did} cell ${cellNo}` + (what ? ` — ${what}` : '') + ran;
    bubble.appendChild(note);

    const pills = document.createElement('div');
    pills.className = 'harness-pills';
    const mkPill = (label: string, onClick: (b: HTMLButtonElement) => Promise<void> | void): HTMLButtonElement => {
      const b = document.createElement('button');
      b.className = 'harness-pill';
      b.textContent = label;
      b.onclick = () => void onClick(b);
      pills.appendChild(b);
      return b;
    };

    const toggle = mkPill('Undo', async b => {
      if (on) {
        revert();
        b.textContent = 'Redo';
      } else {
        await reapply();
        b.textContent = 'Undo';
      }
    });

    mkPill('Copy', b => {
      void navigator.clipboard.writeText(code);
      const prev = b.textContent;
      b.textContent = 'Copied';
      setTimeout(() => (b.textContent = prev), 1000);
    });

    mkPill('New notebook', async b => {
      if (on) {
        revert();
        toggle.textContent = 'Redo';
      }
      await newNotebookWithCode(this.app, code);
      b.disabled = true;
    });

    if (sel) {
      mkPill('Put in selection', b => {
        if (on) {
          revert();
          toggle.textContent = 'Redo';
        }
        sel.cell.model.sharedModel.updateSource(sel.from, sel.to, code);
        b.disabled = true;
      });
    }

    bubble.appendChild(pills);
    this._log.appendChild(bubble);
    this._log.scrollTop = this._log.scrollHeight;
  }

  /** Apply a fix by editing the selection (or the active/errored cell) IN PLACE — never appends. */
  private async _applyFix(
    panel: NotebookPanel,
    code: string,
    sel: { cell: Cell; from: number; to: number } | null,
    targetCell: Cell | null = null
  ): Promise<void> {
    // A named "cell N" wins over the selection/active cell.
    const cell = targetCell ?? (sel ? sel.cell : panel.content.activeCell);
    if (!cell) {
      await this._append('assistant', 'Select the code to fix, or click the cell with the error, then ask again.');
      return;
    }
    const prev = cell.model.sharedModel.getSource();
    const proposed = sel ? prev.slice(0, sel.from) + code + prev.slice(sel.to) : code;
    // Show the whole proposed cell in an editable area — you review/correct, then Apply.
    const fixed = await editCodeDialog(this.rendermime, 'Review the fix — edit & apply', proposed);
    if (fixed === null) {
      await this._append('assistant', '(fix cancelled — nothing changed)');
      return;
    }
    cell.model.sharedModel.setSource(fixed);
    await runCell(panel, cell);

    const bubble = document.createElement('div');
    bubble.className = 'harness-msg harness-assistant';
    const note = document.createElement('div');
    note.className = 'harness-codenote';
    note.textContent = sel ? '✓ Fixed the selected code in place.' : '✓ Fixed the cell in place.';
    bubble.appendChild(note);

    const pills = document.createElement('div');
    pills.className = 'harness-pills';
    const mkPill = (label: string, onClick: (b: HTMLButtonElement) => Promise<void> | void): HTMLButtonElement => {
      const b = document.createElement('button');
      b.className = 'harness-pill';
      b.textContent = label;
      b.onclick = () => void onClick(b);
      pills.appendChild(b);
      return b;
    };
    let on = true;
    mkPill('Undo', async b => {
      if (on) {
        cell.model.sharedModel.setSource(prev);
        on = false;
        b.textContent = 'Redo';
      } else {
        cell.model.sharedModel.setSource(fixed);
        await runCell(panel, cell);
        on = true;
        b.textContent = 'Undo';
      }
    });
    mkPill('Copy', b => {
      void navigator.clipboard.writeText(code);
      const p = b.textContent;
      b.textContent = 'Copied';
      setTimeout(() => (b.textContent = p), 1000);
    });
    bubble.appendChild(pills);
    this._log.appendChild(bubble);
    this._log.scrollTop = this._log.scrollHeight;
  }

  /** After an explanation (e.g. "help"), OFFER to apply a corrected version to the cell the user
   *  asked about: one click writes + runs it (with Undo), or review & edit first. The offer is the
   *  "ask to fix"; the click is the change to the cell. */
  private async _offerFix(panel: NotebookPanel, code: string, cell: Cell): Promise<void> {
    const n = panel.content.widgets.indexOf(cell) + 1;
    const bubble = document.createElement('div');
    bubble.className = 'harness-msg harness-assistant';
    const note = document.createElement('div');
    note.className = 'harness-codenote';
    note.textContent = `I can apply this to Cell ${n} for you.`;
    bubble.appendChild(note);

    const pills = document.createElement('div');
    pills.className = 'harness-pills';
    const prev = cell.model.sharedModel.getSource();
    let applied = false;

    const apply = document.createElement('button');
    apply.className = 'harness-pill';
    apply.textContent = `Apply fix to Cell ${n}`;
    apply.onclick = () => void (async () => {
      if (applied) {
        cell.model.sharedModel.setSource(prev);
        applied = false;
        apply.textContent = `Apply fix to Cell ${n}`;
        this._save();
        return;
      }
      cell.model.sharedModel.setSource(code);
      panel.content.activeCellIndex = panel.content.widgets.indexOf(cell);
      await runCell(panel, cell);
      applied = true;
      apply.textContent = 'Undo';
      this._save();
    })();
    pills.appendChild(apply);

    const review = document.createElement('button');
    review.className = 'harness-pill';
    review.textContent = 'Review & edit';
    review.onclick = () => void (async () => {
      const edited = await editCodeDialog(this.rendermime, `Review the fix — Cell ${n}`, code);
      if (edited === null) {
        return;
      }
      cell.model.sharedModel.setSource(edited);
      panel.content.activeCellIndex = panel.content.widgets.indexOf(cell);
      await runCell(panel, cell);
      applied = true;
      apply.textContent = 'Undo';
      this._save();
    })();
    pills.appendChild(review);

    bubble.appendChild(pills);
    this._log.appendChild(bubble);
    this._log.scrollTop = this._log.scrollHeight;
  }

  private async _append(role: string, content: string): Promise<HTMLElement> {
    const bubble = document.createElement('div');
    bubble.className = 'harness-msg harness-' + role;
    if (role === 'assistant') {
      const r = this.rendermime.createRenderer('text/markdown');
      await r.renderModel(this.rendermime.createModel({ data: { 'text/markdown': content }, trusted: true }));
      bubble.appendChild(r.node);
    } else {
      bubble.textContent = content;
    }
    // A small copy button on every message.
    const copy = document.createElement('button');
    copy.className = 'harness-copy';
    copy.title = 'Copy';
    copy.textContent = '⧉';
    copy.onclick = () => {
      void navigator.clipboard.writeText(content);
      copy.textContent = '✓';
      setTimeout(() => (copy.textContent = '⧉'), 1200);
    };
    bubble.appendChild(copy);
    this._log.appendChild(bubble);
    this._log.scrollTop = this._log.scrollHeight;
    return bubble;
  }

  /** A notebook to borrow a (LIVE) kernel from: the one the user was in, else the focused one, else
   *  any open notebook whose kernel is running. Lets the chat work while an editor/preview is focused
   *  and avoids "No kernel is running" from a stale panel. */
  private _panel(): NotebookPanel | null {
    const order = [this._snapPanel, this.tracker.currentWidget, this._lastPanel];
    for (const p of order) {
      if (p && !p.isDisposed && hasKernel(p)) {
        this._lastPanel = p;
        return p;
      }
    }
    let withKernel: NotebookPanel | null = null;
    let any: NotebookPanel | null = null;
    this.tracker.forEach(p => {
      if (p.isDisposed) {
        return;
      }
      if (!any) {
        any = p;
      }
      if (!withKernel && hasKernel(p)) {
        withKernel = p;
      }
    });
    this._lastPanel = withKernel || any;
    return this._lastPanel;
  }

  private async _submit(): Promise<void> {
    const text = this._input.value.trim();
    if (!text) {
      return;
    }
    this._input.value = '';

    // "make/create/open a new notebook" is a JupyterLab action, not something the model should code.
    if (/^\s*(make|create|new|start|open|add)\s+(me\s+)?(a\s+|an\s+)?(new\s+|blank\s+|fresh\s+)?notebook\b/i.test(text)
        && !/\bcell\b/i.test(text)) {
      this._messages.push({ role: 'user', content: text });
      await this._append('user', text);
      try {
        await this.app.commands.execute('notebook:create-new');
        await this._append('assistant', '✓ Created a new notebook.');
      } catch (e: any) {
        await this._append('assistant', 'Could not create a notebook: ' + e.message);
      }
      this._save();
      return;
    }

    const panel = this._panel();

    // If an image document is the active tab, chat ABOUT the image (vision model).
    const imagePath = this._currentImagePath();
    if (imagePath) {
      this._messages.push({ role: 'user', content: text });
      await this._append('user', text + `\n\n🖼 about ${imagePath}`);
      if (!panel) {
        await this._append('assistant', 'Open a notebook too, so I have a kernel to run the vision model.');
        return;
      }
      this._send.disabled = true;
      const pending = await this._append('assistant', 'looking at the image…');
      try {
        const answer = await kernelEdit(panel, 'vision', { image: imagePath, question: text }, (s: string) => {
          pending.textContent = s;
        });
        pending.remove();
        this._messages.push({ role: 'assistant', content: answer });
        await this._append('assistant', answer);
        this._save();
      } catch (e: any) {
        pending.remove();
        await this._append('assistant', 'Error: ' + e.message);
      } finally {
        this._send.disabled = false;
      }
      return;
    }

    if (!panel) {
      this._messages.push({ role: 'user', content: text });
      await this._append('user', text);
      await this._append('assistant', 'Open a notebook first — I use its kernel.');
      return;
    }

    // What the user is looking at — snapshotted when they reached for the chat (before the cell
    // blurred), so "what's wrong with my code?" sees the focused cell even on a borrowed kernel.
    let context = '';
    let sel = this._lastSel;
    this._lastSel = null;
    // The cell the user was on: the snapshot, else the borrowed panel's active cell.
    const active: Cell | null =
      (this._snapCell && !this._snapCell.isDisposed ? this._snapCell : null) || panel.content.activeCell;
    this._snapCell = null;
    // "remove from cell 3" / "edit cell 2" → target THAT cell by position (1-based, matching the
    // "Cell N" labels the harness prints), whatever is currently selected.
    const cellNumMatch = text.match(/\bcell\s*#?\s*(\d+)\b/i);
    let targetCell: Cell | null = null;
    if (cellNumMatch) {
      const idx = parseInt(cellNumMatch[1], 10) - 1;
      if (idx >= 0 && idx < panel.content.widgets.length) {
        targetCell = panel.content.widgets[idx];
      }
    }
    if (!sel && active) {
      const r = selectionOf(active);
      if (r.scope === 'selection') {
        sel = { cell: active, from: r.from, to: r.to };
      }
    }
    if (sel) {
      const t = sel.cell.model.sharedModel.getSource().slice(sel.from, sel.to);
      if (t.trim()) {
        context = t;
      } else {
        sel = null;
      }
    }
    // The active cell's output/error, so the model can help fix problems.
    const output = active ? cellOutputText(active) : '';
    // Three modes: a QUESTION explains (no cell); a FIX edits in place (no new cell); only a
    // CREATE/build/add request writes a NEW cell. Debugging never spams cells.
    // Modifying EXISTING code (fix it, remove/strip part of it, simplify it) edits the cell in
    // place — it must not spawn a new cell.
    // Prefix patterns (\b only at the START) so remov→remove/removing, delet→delete/deleting, etc.
    const isFix = /\b(fix|debug|edit|updat|chang|refactor|rewrit|modif|correct|repair|solve|remov|delet|get rid|take out|strip|clean ?up|simplif|tidy|drop|without|take away|get rid of)/i.test(text);
    // An explicit "in a cell" / "new cell" means they want a CELL — honour it even over a question.
    const wantsCell = /\b(in|into|as|to)\s+a?\s*(new\s+)?cell\b/i.test(text) || /\bnew cell\b/i.test(text);
    const hasError = /error|traceback/i.test(output);
    const isQuestion =
      /\b(what|why|how|explain|mean|describe|tell me|overview|who|when|where|difference|compare|summar)\b/i.test(text) ||
      /\b(i (?:want|'?d like|would like) to know|know more about|learn about|more about|curious about|info(?:rmation)? (?:on|about))\b/i.test(text) ||
      text.trim().endsWith('?');
    // A bare "help" (no specific build topic) means: look at the cell they're on and explain it /
    // flag problems — never spawn a new cell. "help me write/draw/build X" has its own topic, so it
    // falls through to normal routing.
    const isHelp = /^\s*(?:please\s+|pls\s+|plz\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?h[ae]lp\b/i.test(text);
    const helpHasTopic = /\bhelp\b[^.?!]*\b(write|writing|create|creating|build|building|make|making|generat|add|draw|drawing|implement|example|function|script|program|app)\b/i.test(text)
      && !/\b(this|that|here|above|below|current|selected|my code|the cell|this cell)\b/i.test(text);
    const bareHelp = isHelp && !helpHasTopic;
    // A QUESTION explains (no cell); a FIX edits in place; anything else is a request for CODE, so
    // write it into a cell (a bare 'christmas tree' should produce a cell, not a chat-only answer).
    // A first guess for cell-context attachment; the KERNEL re-classifies intent (NLP) and tells us
    // how to render via a @@HARNESS_ASK@@/@@HARNESS_CODE@@ prefix, which overrides this below.
    let mode: 'ask' | 'fix' | 'create' =
      bareHelp ? 'ask'
      : isFix ? 'fix'
      : wantsCell ? 'create'
      : hasError && !isQuestion ? 'fix'
      : isQuestion ? 'ask'
      : 'create';

    // Attach the active cell's code ONLY when the request is about existing code (fix/ask/error) —
    // NOT for a 'create' request to write something new, where it just distracts the model.
    if (!context && active && mode !== 'create') {
      context = active.model.sharedModel.getSource();
    }
    // A named "cell N" overrides the selection: act on THAT cell and feed its code as context
    // (mode is already 'fix' for an edit verb, 'ask' for a question about it).
    if (targetCell) {
      context = targetCell.model.sharedModel.getSource();
      sel = null;
    }

    // Bare "help" with no cell to look at: ask which cell rather than inventing an answer.
    if (bareHelp && !(context && context.trim())) {
      this._messages.push({ role: 'user', content: text });
      await this._append('user', text);
      const q = "Happy to help — which cell? Click the cell you want help with (or tell me what you're stuck on) and I'll take a look.";
      await this._append('assistant', q);
      this._messages.push({ role: 'assistant', content: q });
      return;
    }

    this._messages.push({ role: 'user', content: text });
    await this._append('user', text);
    // A terse note of what it's looking at — only when it's actually working on an existing cell.
    const ctxCell: Cell | null = targetCell ?? (sel ? sel.cell : active);
    if (ctxCell && context && context.trim() && (sel || mode === 'fix' || hasError || bareHelp)) {
      const n = panel.content.widgets.indexOf(ctxCell) + 1;
      await this._append('context', `Taking a look at Cell ${n}${hasError ? ' — fixing the error' : ''}`);
    }

    this._send.disabled = true;
    this._stop.style.display = '';
    this._busyPanel = panel;
    // A tidy "working" bubble: a spinner + the current status, and a calm streaming body below —
    // instead of one bubble flipping between status text and a jittery raw-token tail (very visible
    // on a slow machine). The status reads the phase; the body fills in as the answer streams.
    const pending = document.createElement('div');
    pending.className = 'harness-msg harness-assistant harness-working';
    const head = document.createElement('div');
    head.className = 'harness-working-head';
    const spin = document.createElement('span');
    spin.className = 'harness-spinner';
    const label = document.createElement('span');
    label.className = 'harness-status';
    label.textContent = 'Thinking…';
    head.append(spin, label);
    const body = document.createElement('div');
    body.className = 'harness-stream';
    pending.append(head, body);
    this._log.appendChild(pending);
    this._log.scrollTop = this._log.scrollHeight;
    let streamed = '';
    try {
      let reply = await kernelEdit(
        panel,
        'chat',
        {
          messages: this._messages,
          context,
          notebook: this._notebookCode(panel),
          open_files: this._openFiles(panel),
          preview_urls: openPreviewUrls(this.app),
          output,
          intent: mode === 'ask' ? 'ask' : 'code',
          think: this._thinkMode
        },
        (s: string) => {
          if (s && s.trim()) {
            label.textContent = s.trim();
          }
        },
        (delta: string) => {
          streamed += delta;
          // While the model is inside <think>…</think>, keep the status reading "thinking" and show
          // nothing raw; once the answer starts, stream it calmly into the body.
          const thinking = streamed.includes('<think>') && !streamed.includes('</think>');
          if (thinking) {
            if (!/thinking/i.test(label.textContent || '')) {
              label.textContent = '💭 thinking…';
            }
            return;
          }
          const answer = streamed.includes('</think>')
            ? (streamed.split('</think>').pop() as string)
            : streamed.replace('<think>', '');
          const shown = answer.trim();
          if (shown) {
            body.textContent = shown.length > 2000 ? '…' + shown.slice(-2000) : shown;
            this._log.scrollTop = this._log.scrollHeight;
          }
        }
      );
      // The kernel classified intent (NLP) and tells us how to render — this overrides our ask-vs-
      // create guess, but NOT an explicit FIX (editing existing code stays an in-place edit).
      if (reply.startsWith('@@HARNESS_ASK@@\n')) {
        if (mode !== 'fix') {
          mode = 'ask';
        }
        reply = reply.slice('@@HARNESS_ASK@@\n'.length);
      } else if (reply.startsWith('@@HARNESS_CODE@@\n')) {
        if (mode !== 'fix') {
          mode = 'create';
        }
        reply = reply.slice('@@HARNESS_CODE@@\n'.length);
      }
      this._messages.push({ role: 'assistant', content: reply });
      pending.remove();
      // A new Tina4 app was provisioned + served: open the App Preview tab at its URL.
      if (reply.startsWith(APP_SENTINEL)) {
        const nl = reply.indexOf('\n');
        const url = reply.slice(0, nl).split('|')[1];
        await this._append('assistant', reply.slice(nl + 1));
        if (url) {
          openAppPreview(this.app, url.trim());
        }
        this._save();
        this._send.disabled = false;
        return;
      }
      // A Tina4 file write: show the code (read-only), DON'T touch the notebook, refresh the preview.
      if (reply.startsWith(WROTE_SENTINEL)) {
        const nl = reply.indexOf('\n');
        const body = nl >= 0 ? reply.slice(nl + 1) : reply;
        await this._append('assistant', body); // note + ```python``` rendered read-only
        const n = reloadPreviews(this.app);
        if (n > 0) {
          await this._append('assistant', `↻ Refreshed ${n} App Preview tab${n > 1 ? 's' : ''}.`);
        }
        this._save();
        this._send.disabled = false;
        return;
      }
      // An example/call: add a NEW cell that calls existing code and RUN it (unless it reads stdin).
      if (reply.startsWith(RUN_SENTINEL)) {
        const { prose, blocks } = splitCode(reply.slice(RUN_SENTINEL.length));
        for (const code of blocks) {
          const cell = insertBelowWithSource(panel, code);
          panel.content.activeCellIndex = panel.content.widgets.indexOf(cell);
          const readsInput = code.split('\n').some(l => /\binput\s*\(/.test(l) && !/^\s/.test(l) && !l.trimStart().startsWith('#'));
          if (!readsInput) {
            await runCell(panel, cell);
          }
        }
        await this._append('assistant', meaningfulProse(prose) || 'Added a cell.');
        this._save();
        this._send.disabled = false;
        return;
      }
      // A "load <file>" request: drop the file straight into a new cell — no pills, any mode.
      if (reply.startsWith(INSERT_SENTINEL)) {
        const { prose, blocks } = splitCode(reply.slice(INSERT_SENTINEL.length));
        for (const code of blocks) {
          const cell = insertBelowWithSource(panel, code);
          panel.content.activeCellIndex = panel.content.widgets.indexOf(cell);
        }
        await this._append('assistant', meaningfulProse(prose) || 'Loaded into a cell.');
        this._save();
        this._send.disabled = false;
        return;
      }
      // A clarifying question with options → show the lead-in and a clickable pill per option; a
      // click sends that choice as the next message (chips, not retyping).
      if (reply.startsWith(CHOICES_SENTINEL)) {
        let data: { lead?: string; options?: string[] } = {};
        try {
          data = JSON.parse(reply.slice(CHOICES_SENTINEL.length));
        } catch {
          data = {};
        }
        const opts = (data.options || []).filter(o => o && o.trim());
        // Replace the raw sentinel we pushed into history with a readable clarification, so the
        // follow-up turn sees clean context.
        if (this._messages.length) {
          this._messages[this._messages.length - 1].content =
            (data.lead ? data.lead + '\n' : '') + opts.map(o => '- ' + o).join('\n');
        }
        if (data.lead) {
          await this._append('assistant', data.lead);
        }
        if (opts.length) {
          const bubble = document.createElement('div');
          bubble.className = 'harness-msg harness-assistant';
          const pills = document.createElement('div');
          pills.className = 'harness-pills';
          for (const opt of opts) {
            const b = document.createElement('button');
            b.className = 'harness-pill';
            b.textContent = opt;
            b.onclick = () => {
              pills.querySelectorAll('button').forEach(x => ((x as HTMLButtonElement).disabled = true));
              this._snapCell = null;      // a follow-up choice, not about a cell
              this._lastSel = null;
              this._input.value = opt;
              void this._submit();
            };
            pills.appendChild(b);
          }
          bubble.appendChild(pills);
          this._log.appendChild(bubble);
          this._log.scrollTop = this._log.scrollHeight;
        }
        this._save();
        this._send.disabled = false;
        return;
      }
      // A VALIDATED fix (kernel decided): show the explanation, then offer to apply the trailing
      // block to the cell. This is the ONLY path that offers a fix.
      if (reply.startsWith(FIX_SENTINEL)) {
        const { prose, blocks } = splitCode(reply.slice(FIX_SENTINEL.length));
        const mp = meaningfulProse(prose);
        if (mp) {
          await this._append('assistant', mp);
        }
        const fixCell: Cell | null = sel ? sel.cell : active;
        const code = blocks.length ? blocks[blocks.length - 1] : '';
        if (fixCell && code) {
          await this._offerFix(panel, code, fixCell);
        }
        this._save();
        this._send.disabled = false;
        return;
      }
      if (mode === 'ask') {
        // A question or a plain "help": show the explanation in the chat, insert nothing, and NEVER
        // offer a fix — only a FIX_SENTINEL (a validated fix) does that. An explanation that happens
        // to echo the code must not turn into an Apply button.
        await this._append('assistant', meaningfulProse(reply) || '(no reply)');
      } else {
        const { prose, blocks } = splitCode(reply);
        const mp = meaningfulProse(prose);
        if (mp) {
          await this._append('assistant', mp);
        }
        for (const code of blocks) {
          if (mode === 'fix') {
            await this._applyFix(panel, code, sel, targetCell);   // edit in place — never append
          } else {
            await this._offerCode(panel, code, sel, false);
          }
        }
        if (!mp && !blocks.length) {
          await this._append('assistant', '(no reply)');
        }
      }
      this._save();
    } catch (e: any) {
      pending.remove();
      await this._append('assistant', 'Error: ' + e.message);
    } finally {
      this._send.disabled = false;
      this._stop.style.display = 'none';
      this._busyPanel = null;
    }
  }

  private _renderThinkBtn(): void {
    const label = { full: 'Think: Full', brief: 'Think: Brief', off: 'Think: Off' }[this._thinkMode];
    this._thinkBtn.textContent = label;
    this._thinkBtn.dataset.mode = this._thinkMode;
  }

  private _cycleThink(): void {
    const order: Array<'full' | 'brief' | 'off'> = ['full', 'brief', 'off'];
    this._thinkMode = order[(order.indexOf(this._thinkMode) + 1) % order.length];
    this._renderThinkBtn();
  }

  /** Stop the running generation by interrupting the kernel (lab.py returns the partial text). */
  private _stopGeneration(): void {
    const kernel = this._busyPanel?.sessionContext.session?.kernel;
    if (kernel) {
      void kernel.interrupt().catch(() => undefined);
    }
    this._stop.disabled = true;
    this._stop.textContent = 'Stopping…';
    window.setTimeout(() => {
      this._stop.disabled = false;
      this._stop.textContent = 'Stop';
    }, 1500);
  }
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'tina4-harness-lab:plugin',
  autoStart: true,
  requires: [INotebookTracker, IRenderMimeRegistry],
  optional: [ICommandPalette],
  activate: (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    rendermime: IRenderMimeRegistry,
    palette: ICommandPalette | null
  ): void => {
    const current = (): NotebookPanel | null => tracker.currentWidget;

    // ---- Chat side panel --------------------------------------------------
    const chat = new ChatPanel(app, tracker, rendermime);
    app.shell.add(chat, 'right', { rank: 900 });
    app.commands.addCommand(CMD.chat, {
      label: 'Harness: Open Chat',
      icon: chatIcon,
      execute: () => {
        app.shell.activateById(chat.id);
      }
    });

    // ---- Check Code -------------------------------------------------------
    app.commands.addCommand(CMD.check, {
      label: 'Harness: Check Code',
      execute: async () => {
        const panel = current();
        const cell = panel?.content.activeCell;
        if (!panel || !cell) {
          return;
        }
        const code = cell.model.sharedModel.getSource();
        const id = Notification.emit('Harness: checking code…', 'in-progress');
        try {
          const review = await kernelEdit(panel, 'check', { code, notebook: collectNotebookCode(panel) });
          Notification.dismiss(id);
          // Show the review AND an editable copy of the cell — Apply writes it back and runs.
          const edited = await editCodeDialog(rendermime, 'Code review — edit & apply', code, review);
          if (edited !== null && edited !== code) {
            cell.model.sharedModel.setSource(edited);
            await NotebookActions.run(panel.content, panel.sessionContext);
          }
        } catch (e: any) {
          Notification.update({ id, message: `Harness: ${e.message}`, type: 'error', autoClose: 5000 });
        }
      }
    });

    // ---- Create Cell ------------------------------------------------------
    app.commands.addCommand(CMD.create, {
      label: 'Harness: Create Cell…',
      execute: async () => {
        const panel = current();
        if (!panel) {
          return;
        }
        const prompt = await promptText(
          'Create a cell',
          'Describe the cell, e.g. "a function that looks up a Star Wars character by id from SWAPI and returns the name, then call it for id 1"'
        );
        if (!prompt) {
          return;
        }
        const id = Notification.emit('Harness: writing a cell…', 'in-progress');
        let code: string;
        try {
          code = await kernelEdit(panel, 'create', { prompt });
          Notification.dismiss(id);
        } catch (e: any) {
          Notification.update({ id, message: `Harness: ${e.message}`, type: 'error', autoClose: 5000 });
          return;
        }
        const choice = await proposeCode(rendermime, 'Suggested cell', code, false);
        if (choice === 'insert') {
          insertBelowWithSource(panel, code);
          await NotebookActions.run(panel.content, panel.sessionContext);
        }
      }
    });

    // ---- Fix Code (selection, or whole cell) ------------------------------
    app.commands.addCommand(CMD.fix, {
      label: 'Harness: Fix Code (selection)',
      execute: async () => {
        const panel = current();
        const cell = panel?.content.activeCell;
        if (!panel || !cell) {
          return;
        }
        const src = cell.model.sharedModel.getSource();
        const { from, to, scope } = selectionOf(cell);
        const target = src.slice(from, to);
        const instruction = await promptText(
          `Fix ${scope}`,
          'What should change? e.g. "add error handling and a docstring"',
          'Suggest'
        );
        if (!instruction) {
          return;
        }
        const id = Notification.emit(`Harness: fixing ${scope}…`, 'in-progress');
        let fixed: string;
        try {
          fixed = await kernelEdit(panel, 'fix', { code: target, instruction });
          Notification.dismiss(id);
        } catch (e: any) {
          Notification.update({ id, message: `Harness: ${e.message}`, type: 'error', autoClose: 5000 });
          return;
        }
        // Editable area so you can correct the suggestion before it touches your cell.
        const edited = await editCodeDialog(rendermime, `Fix ${scope} — edit & apply`, fixed);
        if (edited !== null) {
          if (scope === 'selection') {
            cell.model.sharedModel.updateSource(from, to, edited);
          } else {
            cell.model.sharedModel.setSource(edited);
          }
          await NotebookActions.run(panel.content, panel.sessionContext);
        }
      }
    });

    // ---- Generate here: the CELL'S OWN TEXT is the prompt; replace it with code ----
    app.commands.addCommand(CMD.generate, {
      label: 'Harness: Generate from this cell (replace)',
      execute: async () => {
        const panel = current();
        const cell = panel?.content.activeCell;
        if (!panel || !cell) {
          return;
        }
        const prompt = cell.model.sharedModel.getSource().trim();
        if (!prompt) {
          await promptText('Nothing to generate', 'Type what you want in the cell first, then run this.');
          return;
        }
        const id = Notification.emit('Harness: generating from your prompt…', 'in-progress');
        let code: string;
        try {
          code = await kernelEdit(panel, 'create', { prompt });
          Notification.dismiss(id);
        } catch (e: any) {
          Notification.update({ id, message: `Harness: ${e.message}`, type: 'error', autoClose: 5000 });
          return;
        }
        // Replace the prompt text in the SAME cell with the generated code (editable first).
        const edited = await editCodeDialog(rendermime, 'Replace cell with this code — edit & apply', code);
        if (edited !== null) {
          cell.model.sharedModel.setSource(edited);
          // Only run if it won't block on input() at top level.
          const readsInput = edited.split('\n').some(l => /\binput\s*\(/.test(l) && !/^\s/.test(l) && !l.trimStart().startsWith('#'));
          if (!readsInput) {
            await NotebookActions.run(panel.content, panel.sessionContext);
          }
        }
      }
    });

    // ---- App Preview: open a running app (e.g. a tina4 server) as an IFRAME TAB in JupyterLab ----
    app.commands.addCommand(CMD.preview, {
      label: 'Harness: Open App Preview…',
      execute: async () => {
        const url = await promptText(
          'App preview',
          'URL of the running app to show in a JupyterLab tab',
          'Open',
          'http://localhost:7146/'
        );
        if (!url) {
          return;
        }
        openAppPreview(app, url.trim());
      }
    });

    // ---- Context menu + palette ------------------------------------------
    // All harness actions live in their OWN "Harness" submenu in the right-click menu.
    const selector = '.jp-Notebook .jp-Cell';
    const harnessMenu = new Menu({ commands: app.commands });
    harnessMenu.title.label = 'Harness';
    harnessMenu.addItem({ command: CMD.generate });
    harnessMenu.addItem({ command: CMD.check });
    harnessMenu.addItem({ command: CMD.create });
    harnessMenu.addItem({ command: CMD.fix });
    harnessMenu.addItem({ type: 'separator' });
    harnessMenu.addItem({ command: CMD.preview });
    harnessMenu.addItem({ command: CMD.chat });
    app.contextMenu.addItem({ type: 'separator', selector, rank: 0 });
    app.contextMenu.addItem({ type: 'submenu', submenu: harnessMenu, selector, rank: 0.1 });
    app.contextMenu.addItem({ type: 'separator', selector, rank: 0.2 });

    if (palette) {
      [CMD.generate, CMD.check, CMD.create, CMD.fix, CMD.preview, CMD.chat].forEach(command =>
        palette.addItem({ command, category: 'Harness' })
      );
    }

    // ---- Auto-run the hidden setup for clean demo notebooks ---------------
    // A notebook with metadata {"harness": {"autostart": true}} runs its cells tagged
    // "startup" once, when it opens — so the demo surface starts ready, with setup hidden.
    const started = new WeakSet<NotebookPanel>();
    tracker.widgetAdded.connect((_s, panel) => {
      void panel.sessionContext.ready.then(async () => {
        if (started.has(panel)) {
          return;
        }
        const model: any = panel.content.model;
        const meta = model?.getMetadata ? model.getMetadata('harness') : undefined;
        if (!meta || !meta.autostart) {
          return;
        }
        started.add(panel);
        const nb = panel.content;
        for (let i = 0; i < nb.widgets.length; i++) {
          const tags = (nb.widgets[i].model.getMetadata('tags') as string[]) || [];
          if (tags.includes('startup')) {
            nb.activeCellIndex = i;
            await NotebookActions.run(nb, panel.sessionContext);
          }
        }
      });
    });

    console.log('tina4-harness-lab activated: right-click a cell for Check / Create / Fix.');
  }
};

export default plugin;
