import { listAdxFilterPresets, saveAdxFilterPreset } from "./db.js";

export function createAdxFilterPresets(readSettings, applySettings) {
  const select = document.getElementById("adx-preset");
  const apply = document.getElementById("adx-apply-preset");
  const save = document.getElementById("adx-save-preset");
  const status = document.getElementById("adx-preset-status");
  const dialog = document.getElementById("adx-save-preset-dialog");
  const form = document.getElementById("adx-save-preset-form");
  const nameInput = document.getElementById("adx-preset-name");
  const error = document.getElementById("adx-preset-error");
  const cancel = document.getElementById("adx-cancel-preset");
  const submit = document.getElementById("adx-confirm-preset");
  const presets = new Map();

  function showPresets(selected = "") {
    const options = [...presets.values()].sort((a, b) => a.name.localeCompare(b.name));
    select.replaceChildren(
      new Option(options.length ? "Choose a preset" : "No saved presets", ""),
      ...options.map(preset => new Option(preset.name, preset.key)),
    );
    select.value = selected;
    select.disabled = options.length === 0;
    apply.disabled = !presets.has(select.value);
  }

  listAdxFilterPresets().then(entries => {
    for (const preset of entries) presets.set(preset.key, preset);
    showPresets(select.value);
  }).catch(() => {
    status.textContent = "Saved filter presets could not be loaded.";
  });

  select.addEventListener("change", () => { apply.disabled = !presets.has(select.value); });
  apply.addEventListener("click", () => {
    const preset = presets.get(select.value);
    if (!preset) return;
    applySettings(preset.settings);
    status.textContent = `Applied “${preset.name}”.`;
  });

  save.addEventListener("click", () => {
    form.reset();
    error.textContent = "";
    dialog.showModal();
  });
  cancel.addEventListener("click", () => dialog.close());
  dialog.addEventListener("cancel", event => {
    if (submit.disabled) event.preventDefault();
  });
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (submit.disabled) return;
    const name = nameInput.value.trim();
    if (!name) {
      error.textContent = "Enter a preset name.";
      nameInput.focus();
      return;
    }
    const preset = { key: name.toLowerCase(), name, settings: readSettings() };
    submit.disabled = cancel.disabled = true;
    try {
      await saveAdxFilterPreset(preset);
      presets.set(preset.key, preset);
      showPresets(preset.key);
      dialog.close();
      status.textContent = `Saved “${name}”.`;
    } catch (failure) {
      error.textContent = failure?.name === "ConstraintError"
        ? "That name is already in use. Enter a different name."
        : "The preset could not be saved to this browser.";
      nameInput.select();
      nameInput.focus();
    } finally {
      submit.disabled = cancel.disabled = false;
    }
  });
}
