// Retort schema -> provider JSON schema, strict-mode conversion and local validation.
// Covers: structured schema (toJSON with array bounds), provider schema
// (compact repeated enums into $defs) and model transport helpers (strictSchema, matches, mismatch,
// decodeStrictOptionals). No string length limits are added anywhere: model output is never capped.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { retortSchemaToJsonSchema } = require("retort-js/dist/tooling");

export function toJSON(parameters) {
  const schema = retortSchemaToJsonSchema(parameters);
  function bounds(source, target) {
    if (!source || !target || typeof source !== "object") return;
    if (!Array.isArray(source) && Object.hasOwn(source, "type")) {
      for (const key of ["minItems", "maxItems"]) if (source[key] !== undefined) target[key] = source[key];
      source = source.type;
    }
    if (Array.isArray(source)) bounds(source[0], target.items);
    else if (source && typeof source === "object")
      for (const [key, value] of Object.entries(source)) bounds(value, target.properties?.[key]);
  }
  bounds(parameters, schema);
  return schema;
}

export function strictSchema(schema) {
  const out = { ...schema };
  if (schema.anyOf) out.anyOf = schema.anyOf.map(strictSchema);
  if (schema.type === "object") {
    if (!schema.properties) throw Error("Strict Retort objects require typed properties.");
    out.properties = Object.fromEntries(Object.entries(schema.properties).map(([k, child]) => [k, strictSchema(child)]));
    const required = new Set(schema.required || []);
    for (const [key, child] of Object.entries(out.properties)) if (!required.has(key)) {
      if (child.anyOf) child.anyOf = [...child.anyOf, { type: "null" }];
      else child.type = [...new Set([...(Array.isArray(child.type) ? child.type : [child.type]), "null"])];
      if (child.enum && !child.enum.includes(null)) child.enum = [...child.enum, null];
    }
    out.required = Object.keys(out.properties);
    out.additionalProperties = false;
  }
  if (schema.type === "array") {
    if (!schema.items) throw Error("Strict Retort arrays require a typed item schema.");
    out.items = strictSchema(schema.items);
  }
  return out;
}

// Reuse identical enum schemas through $defs (providers count enum values across the whole schema).
export function compact(schema) {
  const counts = new Map(), definitions = {}, names = new Map();
  const children = (node, visit) => {
    for (const child of Object.values(node.properties || {})) visit(child);
    if (node.items) visit(node.items);
    for (const child of node.anyOf || []) visit(child);
  };
  const count = (node) => {
    if (node.enum) { const key = JSON.stringify(node); counts.set(key, (counts.get(key) || 0) + 1); }
    children(node, count);
  };
  count(schema);
  const encode = (node) => {
    const key = JSON.stringify(node);
    if (node.enum && counts.get(key) > 1) {
      if (!names.has(key)) { const name = "enum" + names.size; names.set(key, name); definitions[name] = structuredClone(node); }
      return { $ref: "#/$defs/" + names.get(key) };
    }
    const out = { ...node };
    if (node.properties) out.properties = Object.fromEntries(Object.entries(node.properties).map(([k, v]) => [k, encode(v)]));
    if (node.items) out.items = encode(node.items);
    if (node.anyOf) out.anyOf = node.anyOf.map(encode);
    return out;
  };
  const result = encode(schema);
  if (Object.keys(definitions).length) result.$defs = definitions;
  return result;
}

const typeOf = (value) => (Array.isArray(value) ? "array" : value === null ? "null" : Number.isInteger(value) ? "integer" : typeof value);
const typeMatches = (value, wanted) => {
  const actual = typeOf(value);
  return [].concat(wanted).some((t) => t === actual || (t === "number" && actual === "integer"));
};

export function matches(value, schema) {
  if (schema.anyOf && !schema.anyOf.some((branch) => matches(value, branch))) return false;
  if (schema.type && !typeMatches(value, schema.type)) return false;
  if (schema.enum && !schema.enum.includes(value)) return false;
  if (schema.const !== undefined && schema.const !== value) return false;
  if (typeof value === "number" && (!Number.isFinite(value) || (schema.minimum !== undefined && value < schema.minimum) ||
      (schema.maximum !== undefined && value > schema.maximum))) return false;
  if (typeof value === "string" && schema.pattern && !new RegExp(schema.pattern).test(value)) return false;
  if (Array.isArray(value) && ((schema.minItems !== undefined && value.length < schema.minItems) ||
      (schema.maxItems !== undefined && value.length > schema.maxItems) ||
      (schema.items && !value.every((item) => matches(item, schema.items))))) return false;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    if ((schema.required || []).some((key) => !Object.hasOwn(value, key))) return false;
    if (schema.additionalProperties === false && Object.keys(value).some((key) => !Object.hasOwn(schema.properties || {}, key))) return false;
    for (const [key, child] of Object.entries(schema.properties || {}))
      if (Object.hasOwn(value, key) && !matches(value[key], child)) return false;
  }
  return true;
}

export function mismatch(value, schema, path = "$") {
  if (schema.anyOf) return schema.anyOf.some((b) => matches(value, b)) ? "" : `${path}: no allowed variant matches`;
  if (schema.type && !typeMatches(value, schema.type)) return `${path}: expected ${[].concat(schema.type).join(" or ")}; received ${typeOf(value)}`;
  if (schema.enum && !schema.enum.includes(value)) return `${path}: ${JSON.stringify(value)} is not one of ${schema.enum.join(", ")}`;
  if (typeof value === "number" && !Number.isFinite(value)) return `${path}: not a finite number`;
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) return `${path}: needs at least ${schema.minItems} items`;
    if (schema.maxItems !== undefined && value.length > schema.maxItems) return `${path}: allows at most ${schema.maxItems} items`;
    for (let i = 0; i < value.length; i++) { const m = schema.items && mismatch(value[i], schema.items, `${path}[${i}]`); if (m) return m; }
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const key of schema.required || []) if (!Object.hasOwn(value, key)) return `${path}.${key}: missing`;
    if (schema.additionalProperties === false)
      for (const key of Object.keys(value)) if (!Object.hasOwn(schema.properties || {}, key)) return `${path}.${key}: not allowed`;
    for (const [key, child] of Object.entries(schema.properties || {})) {
      if (!Object.hasOwn(value, key)) continue;
      const m = mismatch(value[key], child, `${path}.${key}`);
      if (m) return m;
    }
  }
  return "";
}

export function decodeStrictOptionals(value, schema) {
  if (Array.isArray(value) && schema.items) { for (const item of value) decodeStrictOptionals(item, schema.items); return value; }
  if (value && typeof value === "object")
    for (const [key, child] of Object.entries(schema.properties || {})) {
      if (value[key] === null && !(schema.required || []).includes(key)) delete value[key];
      else if (Object.hasOwn(value, key)) decodeStrictOptionals(value[key], child);
    }
  return value;
}

// One call: Retort parameters -> { answerSchema (local check), wireSchema (strict, compacted for the provider) }.
export function contract(parameters) {
  const answerSchema = toJSON(parameters);
  return { answerSchema, wireSchema: compact(strictSchema(answerSchema)) };
}
