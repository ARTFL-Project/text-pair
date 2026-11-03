import globalConfig from "./config.js";

// Query string from params, skipping empty values; arrays repeat the key.
export function paramsToUrl(params) {
    const urlParams = [];
    for (const [key, value] of Object.entries(params)) {
        if (value !== "" && value !== null && value !== undefined) {
            if (Array.isArray(value)) {
                value.forEach((val) => {
                    if (val) {
                        urlParams.push(`${encodeURIComponent(key)}=${encodeURIComponent(val)}`);
                    }
                });
            } else {
                urlParams.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
            }
        }
    }
    return urlParams.join("&");
}

// Text of an HTML label, such as sourceLabel or databaseLabel.
export function plainText(html) {
    return new DOMParser().parseFromString(html || "", "text/html").body.textContent;
}

export function prefersReducedMotion() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// A value without the quotes of an exact match, which is how a clicked count is searched.
export function unquote(value) {
    const quoted = /^"(.+)"$/.exec(value);
    return quoted ? quoted[1] : value;
}

// A search field with its value as a phrase: an author "Voltaire", a year "1790–1830", no title.
export function fieldPhrase(label, value) {
    const name = (label || "").toLowerCase();
    if (value == '""') return `no ${name}`;
    const bare = unquote(value);
    const shown = bare == value && /^\d+-\d+$/.test(value) ? value.replace("-", "\u2013") : bare;
    return `${/^[aeiou]/.test(name) ? "an" : "a"} ${name} \u201c${shown}\u201d`;
}

// What a document is called when its citation fields are all empty.
export const UNATTRIBUTED = "Unattributed text";

// The collection of the document on a side of an alignment: its own, unless a passage family put a
// document of the other collection there, which the API marks (two underscores: no field is named so).
export function documentCollection(alignment, side) {
    return alignment[`${side}__collection`] || side;
}

// The search fields ("search") or facets of reused passages where they were first found (side "source") or
// found again ("target"), as { value, label }: each goes by the field of that collection, or one named after
// the other's (web_settings.role_key). An API that serves none has the passage pairs' instead.
export function reusedFields(kind, side) {
    const fields = globalConfig[kind == "search" ? "reusedSearchFields" : "reusedFacets"];
    if (!fields || !fields.length) return globalConfig[kind == "search" ? "metadataFields" : "facetsFields"][side];
    return fields.map((field) => ({
        value: field.fields.find((name) => name.startsWith(`${side}_`)) || `${side}_${field.fields[0].split("_").slice(1).join("_")}`,
        label: field.label,
    }));
}
