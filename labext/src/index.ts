/**
 * tina4-harness-lab — right-click a notebook cell to use the notebook's OWN local model.
 *
 *   • Check Code   — review the active cell's code
 *   • Create Cell  — describe a cell; the model writes it (e.g. a SWAPI JSON lookup) and runs it
 *   • Fix Code     — rewrite the highlighted selection (or the whole cell) from an instruction
 *
 * The extension never ships a model. It calls a tiny Python function, `__harness_edit`, that runs
 * in the notebook's kernel and uses the already-loaded `model` (loading it on first use if needed).
 * So a clean notebook + a local LLM is all you need.
 */
import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { Dialog, ICommandPalette, Notification, showDialog } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { Widget } from '@lumino/widgets';

const CMD = {
  check: 'tina4-harness:check-code',
  create: 'tina4-harness:create-cell',
  fix: 'tina4-harness:fix-code'
};

/** Run `__harness_edit(kind, payload)` in the kernel and return the model's text. */
async function kernelEdit(
  panel: NotebookPanel,
  kind: 'create' | 'check' | 'fix',
  payload: Record<string, unknown>
): Promise<string> {
  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel) {
    throw new Error('No kernel is running for this notebook.');
  }
  const b64 = btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
  // Self-bootstrapping: make `harness` importable even if the notebook's setup cell
  // hasn't run yet (the kernel's cwd is the project root or notebooks/).
  const code =
    'import sys, os\n' +
    "sys.path[:0] = [p for p in ('src', '../src') if os.path.isdir(p) and p not in sys.path]\n" +
    'import harness.lab as _hl\n' +
    `print(_hl.harness_edit(${JSON.stringify(kind)}, "${b64}"))`;
  const future = kernel.requestExecute({
    code,
    stop_on_error: false,
    store_history: false,
    silent: false
  });
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
    throw new Error(
      'No reply from the kernel. Is the harness installed and the kernel running?'
    );
  }
  const line = buf.slice(i + marker.length).split('\n')[0].trim();
  const res = JSON.parse(line);
  if (res.error) {
    throw new Error(res.error as string);
  }
  return res.text as string;
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
    setTimeout(() => this._ta.focus(), 0);
  }
  getValue(): string {
    return this._ta.value;
  }
}

/** A dialog body that shows read-only text in a <pre>. */
class PreBody extends Widget {
  constructor(text: string) {
    super();
    this.addClass('harness-dialog');
    const pre = document.createElement('pre');
    pre.className = 'harness-pre';
    pre.textContent = text;
    this.node.appendChild(pre);
  }
}

async function promptText(
  title: string,
  placeholder: string,
  okLabel = 'Run'
): Promise<string | null> {
  const body = new TextareaBody(placeholder);
  const result = await showDialog({
    title,
    body,
    buttons: [Dialog.cancelButton(), Dialog.okButton({ label: okLabel })]
  });
  if (!result.button.accept) {
    return null;
  }
  const value = body.getValue().trim();
  return value || null;
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'tina4-harness-lab:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette],
  activate: (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    palette: ICommandPalette | null
  ): void => {
    const current = (): NotebookPanel | null => tracker.currentWidget;

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
          await showDialog({
            title: 'Harness — code review',
            body: new PreBody(review),
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
        try {
          const codeText = await kernelEdit(panel, 'create', { prompt });
          NotebookActions.insertBelow(panel.content);
          const cell = panel.content.activeCell!;
          cell.model.sharedModel.setSource(codeText);
          await NotebookActions.run(panel.content, panel.sessionContext);
          Notification.update({ id, message: 'Harness: cell created & run', type: 'success', autoClose: 2500 });
        } catch (e: any) {
          Notification.update({ id, message: `Harness: ${e.message}`, type: 'error', autoClose: 5000 });
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
        const editor = cell.editor;
        const src = cell.model.sharedModel.getSource();
        let from = 0;
        let to = src.length;
        let scope = 'cell';
        if (editor) {
          const sel = editor.getSelection();
          const f = editor.getOffsetAt(sel.start);
          const t = editor.getOffsetAt(sel.end);
          if (t > f) {
            from = f;
            to = t;
            scope = 'selection';
          }
        }
        const selected = src.slice(from, to);
        const instruction = await promptText(
          `Fix ${scope}`,
          'What should change? e.g. "add error handling and a docstring"',
          'Apply'
        );
        if (!instruction) {
          return;
        }
        const id = Notification.emit(`Harness: fixing ${scope}…`, 'in-progress');
        try {
          const fixed = await kernelEdit(panel, 'fix', { code: selected, instruction });
          cell.model.sharedModel.updateSource(from, to, fixed);
          Notification.update({ id, message: `Harness: ${scope} replaced`, type: 'success', autoClose: 2500 });
        } catch (e: any) {
          Notification.update({ id, message: `Harness: ${e.message}`, type: 'error', autoClose: 5000 });
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
      [CMD.check, CMD.create, CMD.fix].forEach(command =>
        palette.addItem({ command, category: 'Harness' })
      );
    }

    console.log('tina4-harness-lab activated: right-click a cell for Check / Create / Fix.');
  }
};

export default plugin;
