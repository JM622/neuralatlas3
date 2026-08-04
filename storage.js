/* storage.js — shared localStorage helpers for NeuralAtlas.
   Everything here is stored ONLY on this device/browser — there is no server. */

const NA_STORAGE_KEY = "neuralatlas_experiments_v1";
const NA_PROFILE_KEY = "neuralatlas_profile_v1";

function naLoadExperiments() {
  try { return JSON.parse(localStorage.getItem(NA_STORAGE_KEY) || "[]"); }
  catch (e) { return []; }
}
function naSaveExperiments(list) {
  try { localStorage.setItem(NA_STORAGE_KEY, JSON.stringify(list)); return true; }
  catch (e) { return false; }
}
function naAddExperiment(exp) {
  const list = naLoadExperiments();
  list.unshift(exp);
  naSaveExperiments(list);
  return list;
}
function naDeleteExperiment(id) {
  const list = naLoadExperiments().filter((e) => e.id !== id);
  naSaveExperiments(list);
  return list;
}
function naRenameExperiment(id, label) {
  const list = naLoadExperiments();
  const e = list.find((x) => x.id === id);
  if (e) e.label = label;
  naSaveExperiments(list);
  return list;
}
function naImportExperiment(expObj) {
  if (!expObj || !expObj.id) expObj = { ...expObj, id: "imp_" + Date.now() + "_" + Math.floor(Math.random() * 1e6) };
  return naAddExperiment(expObj);
}

function naLoadProfile() {
  try { return JSON.parse(localStorage.getItem(NA_PROFILE_KEY) || "null"); }
  catch (e) { return null; }
}
function naSaveProfile(p) {
  try { localStorage.setItem(NA_PROFILE_KEY, JSON.stringify(p)); return true; }
  catch (e) { return false; }
}

function naDownloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function naDownloadCSV(rows, filename) {
  const csv = rows.map((r) => r.map((v) => (typeof v === "string" && v.includes(",")) ? `"${v}"` : v).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function naSlug(s) {
  return (s || "experiment").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}
