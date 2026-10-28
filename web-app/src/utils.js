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

// A search field with its value as a phrase: an author "Voltaire", a year "1790–1830", no title.
export function fieldPhrase(label, value) {
    const name = (label || "").toLowerCase();
    if (value == '""') return `no ${name}`;
    const shown = /^\d+-\d+$/.test(value) ? value.replace("-", "\u2013") : value;
    return `${/^[aeiou]/.test(name) ? "an" : "a"} ${name} \u201c${shown}\u201d`;
}
