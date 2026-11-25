const categories = new Set(["sp500", "adxState", "maStatus", "marketRegime"]);
const dates = new Set(["highDate", "lowDate"]);
const fractions = new Set(["ma20Gap", "ma20GapPercentile", "maPosition", "pct50", "priorPct50", "pct50Change", "minPct50", "maxPct50"]);
const points = new Set(["gapChange", "pct50Change"]);
const numericConditions = [
  ["lte", "At or below"], ["lt", "Below"], ["gte", "At or above"], ["gt", "Above"], ["eq", "Equals"],
];
const dateConditions = [
  ["lte", "On or before"], ["lt", "Before"], ["gte", "On or after"], ["gt", "After"], ["eq", "On"],
];
const textConditions = [["contains", "Contains"], ["eq", "Equals"]];
const relation = { lte: "≤", lt: "<", gte: "≥", gt: ">", eq: "=", contains: "contains" };
const compare = {
  lte: (value, target) => value <= target,
  lt: (value, target) => value < target,
  gte: (value, target) => value >= target,
  gt: (value, target) => value > target,
  eq: (value, target) => value === target,
  contains: (value, target) => value.includes(target),
};

export function describeAdxCondition(field, label, rule) {
  if (rule.values) return `${label} = ${rule.values.map(value => value ?? "(Blanks)").join(" or ")}`;
  const unit = points.has(field) ? " pp" : fractions.has(field) ? "%" : "";
  const description = rule.conditions.map(({ op, value }) => `${relation[op]} ${value}${unit}`).join(` ${rule.join} `);
  return `${label} ${rule.conditions.length > 1 ? `(${description})` : description}`;
}

function select(options, label, selected) {
  const control = document.createElement("select");
  control.setAttribute("aria-label", label);
  control.append(...options.map(([value, text]) => new Option(text, value)));
  control.value = selected;
  return control;
}

function categoryValue(row, field) {
  return row[field] || null;
}

function filterPredicate(field, filter) {
  if (categories.has(field)) {
    const included = new Set(filter.values);
    return row => included.has(categoryValue(row, field));
  }
  const text = field === "ticker" || dates.has(field);
  const comparisons = filter.conditions.map(({ op, value }) => {
    const target = text ? value.toUpperCase() : Number(value);
    return actual => compare[op](actual, target);
  });
  return row => {
    const actual = row[field];
    if (actual == null || actual === "") return false;
    const value = text ? actual.toUpperCase() : fractions.has(field) ? actual * 100 : actual;
    return filter.join === "and" ? comparisons.every(check => check(value)) : comparisons.some(check => check(value));
  };
}

function categoryEditor(body, field, rows, saved, apply) {
  const values = [...new Set([
    ...rows.map(row => categoryValue(row, field)),
    ...(saved?.values ?? []),
  ])].sort((a, b) => (a ?? "").localeCompare(b ?? ""));
  const selected = new Set(saved?.values ?? values);

  function checkbox(text, checked) {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    label.append(input, text);
    body.append(label);
    return input;
  }

  const all = checkbox("Select all", false);
  const items = values.map(value => checkbox(value ?? "(Blanks)", selected.has(value)));
  function sync() {
    const checked = items.filter(item => item.checked).length;
    all.checked = checked > 0 && checked === items.length;
    all.indeterminate = checked > 0 && !all.checked;
    apply.disabled = checked === 0;
  }
  all.disabled = items.length === 0;
  all.addEventListener("change", () => {
    for (const item of items) item.checked = all.checked;
    sync();
  });
  for (const item of items) item.addEventListener("change", sync);
  sync();

  return () => {
    const included = values.filter((_, index) => items[index].checked);
    return included.length === values.length ? null : { values: included };
  };
}

function conditionEditor(body, field, saved) {
  const type = field === "ticker" ? "text" : dates.has(field) ? "date" : "number";
  const conditions = type === "number" ? numericConditions : type === "date" ? dateConditions : textConditions;
  const unit = points.has(field) ? " (pp)" : fractions.has(field) ? " (%)" : "";

  function conditionRow(index) {
    const previous = saved?.conditions[index - 1];
    const row = document.createElement("div");
    row.className = "adx-filter-condition";
    const options = index === 1 ? conditions : [["", "No second condition"], ...conditions];
    const operator = select(options, `Condition ${index}`, previous?.op ?? options[0][0]);
    const value = document.createElement("input");
    value.type = type;
    if (type === "number") value.step = "any";
    value.placeholder = `Value${unit}`;
    value.setAttribute("aria-label", `Value ${index}${unit}`);
    value.value = previous?.value ?? "";
    value.disabled = !operator.value;
    operator.addEventListener("change", () => { value.disabled = !operator.value; });
    row.append(operator, value);
    return { row, operator, value };
  }

  const first = conditionRow(1);
  const second = conditionRow(2);
  const join = select([["and", "And"], ["or", "Or"]], "Combine conditions", saved?.join ?? "and");
  join.className = "adx-filter-join";
  body.append(first.row, join, second.row);
  return () => {
    const terms = [first, second].filter(({ operator, value }) => operator.value && value.value.trim())
      .map(({ operator, value }) => ({ op: operator.value, value: value.value.trim() }));
    return terms.length ? { conditions: terms, join: join.value } : null;
  };
}

export function createAdxColumnFilters(updateResults) {
  const menu = document.getElementById("adx-column-menu");
  const heading = document.getElementById("adx-column-menu-title");
  const body = document.getElementById("adx-column-menu-body");
  const source = document.getElementById("adx-sp500-source");
  const apply = document.getElementById("adx-apply-filter");
  const clear = document.getElementById("adx-clear-filter");
  const clearAll = document.getElementById("adx-clear-filters");
  const scroller = document.getElementById("adx-scroll");
  const filters = new Map();
  const buttons = new Map();
  let rows = [];
  let current;

  function refresh() {
    for (const [field, button] of buttons) button.toggleAttribute("data-filtered", filters.has(field));
    clearAll.disabled = filters.size === 0;
    clearAll.textContent = filters.size ? `Clear column filters (${filters.size})` : "Clear column filters";
    updateResults();
  }

  function close() {
    if (menu.matches(":popover-open")) menu.hidePopover();
  }

  function save(filter) {
    const { field, button } = current;
    if (filter) filters.set(field, filter);
    else filters.delete(field);
    close();
    refresh();
    button.focus({ preventScroll: true });
  }

  function position(button) {
    const anchor = button.getBoundingClientRect();
    const { width, height } = menu.getBoundingClientRect();
    const { clientWidth, clientHeight } = document.documentElement;
    const left = Math.max(8, Math.min(anchor.right - width, clientWidth - width - 8));
    const preferredTop = anchor.bottom + height + 12 <= clientHeight ? anchor.bottom + 4 : anchor.top - height - 4;
    const top = Math.max(8, Math.min(preferredTop, clientHeight - height - 8));
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }

  function reposition() {
    if (current) position(current.button);
  }

  menu.addEventListener("beforetoggle", event => {
    if (!current) return;
    const open = event.newState === "open";
    current.button.setAttribute("aria-expanded", String(open));
    if (!open) current = undefined;
  });
  menu.addEventListener("toggle", event => {
    if (event.newState !== "open" || !current) return;
    position(current.button);
    body.querySelector("select:not(:disabled), input:not(:disabled)")?.focus({ preventScroll: true });
  });
  menu.addEventListener("submit", event => {
    event.preventDefault();
    if (current) save(current.read());
  });
  clear.addEventListener("click", () => { if (current) save(null); });
  clearAll.addEventListener("click", () => {
    close();
    filters.clear();
    refresh();
  });
  scroller.addEventListener("scroll", reposition);
  window.addEventListener("scroll", reposition);
  window.addEventListener("resize", reposition);
  window.addEventListener("hashchange", close);

  return {
    createButton(field, label) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "adx-header-filter";
      button.title = `Filter ${label}`;
      button.setAttribute("aria-label", button.title);
      button.setAttribute("aria-haspopup", "dialog");
      button.setAttribute("aria-expanded", "false");
      button.setAttribute("aria-controls", menu.id);
      button.setAttribute("popovertarget", menu.id);
      button.addEventListener("click", () => {
        if (current?.button === button) return;
        close();
        body.replaceChildren();
        heading.textContent = label;
        source.hidden = field !== "sp500";
        const saved = filters.get(field);
        apply.disabled = false;
        clear.disabled = !saved;
        const read = categories.has(field) ? categoryEditor(body, field, rows, saved, apply) : conditionEditor(body, field, saved);
        current = { field, button, read };
      });
      buttons.set(field, button);
      return button;
    },
    close,
    setRows(values) { rows = values; },
    predicates() { return [...filters].map(([field, filter]) => filterPredicate(field, filter)); },
    snapshot() { return [...filters]; },
    restore(settings) {
      close();
      filters.clear();
      for (const [field, filter] of settings) {
        if (buttons.has(field)) filters.set(field, filter);
      }
      refresh();
    },
  };
}
