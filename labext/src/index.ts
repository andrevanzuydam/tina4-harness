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
import { Widget } from '@lumino/widgets';

const CMD = {
  check: 'tina4-harness:check-code',
  create: 'tina4-harness:create-cell',
  fix: 'tina4-harness:fix-code',
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
  payload: Record<string, unknown>
): Promise<string> {
  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel) {
    throw new Error('No kernel is running for this notebook.');
  }
  const b64 = btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
  const code =
    'import sys, os\n' +
    "sys.path[:0] = [p for p in ('src', '../src') if os.path.isdir(p) and p not in sys.path]\n" +
    'import harness.lab as _hl\n' +
    `print(_hl.harness_edit(${JSON.stringify(kind)}, "${b64}"))`;
  const future = kernel.requestExecute({ code, stop_on_error: false, store_history: false });
  let buf = '';
  future.onIOPub = (msg: any): void => {
    const t = msg.header.msg_type;
    if (t === 'stream' && msg.content.name === 'stdout') {
      buf += msg.content.text as string;
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
  constructor(placeholder: string) {
    super();
    this.addClass('harness-dialog');
    this._ta = document.createElement('textarea');
    this._ta.placeholder = placeholder;
    this._ta.rows = 4;
    this.node.appendChild(this._ta);
    setTimeout(() => this._ta.focus(), 0);
  }
  getValue(): string {
    return this._ta.value;
  }
}

async function promptText(title: string, placeholder: string, okLabel = 'Run'): Promise<string | null> {
  const body = new TextareaBody(placeholder);
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
  private _ctxLabel: HTMLSpanElement;
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
    const clear = document.createElement('button');
    clear.className = 'harness-chat-clear';
    clear.textContent = 'Clear';
    clear.title = 'Clear this conversation';
    clear.onclick = () => this._clear();
    header.appendChild(this._ctxLabel);
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
    row.appendChild(this._input);
    row.appendChild(this._send);
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

    // Conversation follows the active notebook / file.
    const shell: any = this.app.shell;
    if (shell.currentChanged && shell.currentChanged.connect) {
      shell.currentChanged.connect(() => void this._syncContext());
    }
    this.tracker.currentChanged.connect(() => void this._syncContext());

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
    sel: { cell: Cell; from: number; to: number } | null
  ): Promise<void> {
    const placed = placeCode(panel, code);
    await runCell(panel, placed.cell);

    // Track the live cell so Undo/Redo can toggle it.
    let cur: Cell | null = placed.cell;
    let wasCreated = placed.created;
    const removeCode = (): void => {
      if (!cur) {
        return;
      }
      if (wasCreated) {
        deleteCellWidget(panel, cur);
      } else {
        cur.model.sharedModel.setSource('');
      }
      cur = null;
    };

    const bubble = document.createElement('div');
    bubble.className = 'harness-msg harness-assistant';
    const note = document.createElement('div');
    note.className = 'harness-codenote';
    note.textContent = placed.created ? '✓ Added a cell and ran it.' : '✓ Filled the blank cell and ran it.';
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

    // Undo ⇄ Redo toggle (never disabled).
    const toggle = mkPill('Undo', async b => {
      if (cur) {
        removeCode();
        b.textContent = 'Redo';
      } else {
        const r = placeCode(panel, code);
        cur = r.cell;
        wasCreated = r.created;
        await runCell(panel, cur);
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
      removeCode();
      toggle.textContent = 'Redo';
      await newNotebookWithCode(this.app, code);
      b.disabled = true;
    });

    if (sel) {
      mkPill('Put in selection', b => {
        removeCode();
        toggle.textContent = 'Redo';
        sel.cell.model.sharedModel.updateSource(sel.from, sel.to, code);
        b.disabled = true;
      });
    }

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

  private async _submit(): Promise<void> {
    const text = this._input.value.trim();
    if (!text) {
      return;
    }
    this._input.value = '';

    const panel = this.tracker.currentWidget;

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
        const answer = await kernelEdit(panel, 'vision', { image: imagePath, question: text });
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

    // If you've highlighted code in a cell, send it as context so "this"/"it" is understood.
    let context = '';
    let sel: { cell: Cell; from: number; to: number } | null = null;
    const cell = panel.content.activeCell;
    if (cell) {
      const { from, to, scope } = selectionOf(cell);
      if (scope === 'selection') {
        context = cell.model.sharedModel.getSource().slice(from, to);
        sel = { cell, from, to };
      }
    }

    this._messages.push({ role: 'user', content: text });
    await this._append('user', text + (context ? '\n\n↳ (with your selection)' : ''));

    this._send.disabled = true;
    const pending = await this._append('assistant', '…');
    try {
      const reply = await kernelEdit(panel, 'chat', {
        messages: this._messages,
        context,
        notebook: this._notebookCode(panel)
      });
      this._messages.push({ role: 'assistant', content: reply });
      pending.remove();
      // Code is offered with action pills (Insert / Insert & run / Replace) — not auto-run.
      const { prose, blocks } = splitCode(reply);
      if (prose) {
        await this._append('assistant', prose);
      }
      for (const code of blocks) {
        await this._offerCode(panel, code, sel);
      }
      if (!prose && !blocks.length) {
        await this._append('assistant', '(no reply)');
      }
      this._save();
    } catch (e: any) {
      pending.remove();
      await this._append('assistant', 'Error: ' + e.message);
    } finally {
      this._send.disabled = false;
    }
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
          const review = await kernelEdit(panel, 'check', { code });
          Notification.dismiss(id);
          const body = await markdownWidget(rendermime, review);
          await showDialog({
            title: 'Harness — code review',
            body,
            buttons: [Dialog.okButton({ label: 'Close' })]
          });
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
        const choice = await proposeCode(rendermime, `Suggested fix (${scope})`, fixed, true);
        if (choice === 'replace') {
          cell.model.sharedModel.updateSource(from, to, fixed);
        } else if (choice === 'insert') {
          insertBelowWithSource(panel, fixed);
        }
      }
    });

    // ---- Context menu + palette ------------------------------------------
    const selector = '.jp-Notebook .jp-Cell';
    app.contextMenu.addItem({ type: 'separator', selector, rank: 0 });
    app.contextMenu.addItem({ command: CMD.check, selector, rank: 1 });
    app.contextMenu.addItem({ command: CMD.create, selector, rank: 2 });
    app.contextMenu.addItem({ command: CMD.fix, selector, rank: 3 });
    app.contextMenu.addItem({ type: 'separator', selector, rank: 4 });

    if (palette) {
      [CMD.check, CMD.create, CMD.fix, CMD.chat].forEach(command =>
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
