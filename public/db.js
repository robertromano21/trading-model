const LOG_LIMIT = 5000;
const database = new Promise((resolve, reject) => {
  const request = indexedDB.open("TradingAppDB", 5);
  request.onupgradeneeded = (event) => {
    const db = request.result;
    if (event.oldVersion < 1) {
      db.createObjectStore("marketData", { keyPath: ["ticker", "date"] }).createIndex("date", "date");
      db.createObjectStore("browserLogs", { keyPath: "id", autoIncrement: true }).createIndex("timestamp", "timestamp");
    }
    if (event.oldVersion < 2) db.createObjectStore("modelResults");
    if (event.oldVersion < 3) {
      db.createObjectStore("backtestRuns", { keyPath: "runId" }).createIndex("createdAt", "createdAt");
    }
    if (event.oldVersion < 4) db.createObjectStore("adxFilterPresets", { keyPath: "key" });
    if (event.oldVersion < 5) db.createObjectStore("paperTrades", { keyPath: ["ticker", "entryDate"] });
  };
  request.onsuccess = () => {
    request.result.onversionchange = () => request.result.close();
    resolve(request.result);
  };
  request.onerror = () => reject(request.error);
  request.onblocked = () => reject(new Error("Close other dashboard tabs and reload to open local storage."));
});

async function transaction(storeName, mode, work) {
  const db = await database;
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, mode);
    const store = tx.objectStore(storeName);
    tx.oncomplete = () => resolve(result);
    tx.onabort = () => reject(tx.error || new Error("IndexedDB transaction was aborted."));
    let result;
    try {
      work(store, (value) => { result = value; });
    } catch (error) {
      reject(error);
      tx.abort();
    }
  });
}

export function cacheSession(date, bars) {
  return transaction("marketData", "readwrite", (store) => {
    const keys = store.index("date").getAllKeys(date);
    keys.onsuccess = () => {
      for (const key of keys.result) store.delete(key);
      for (const bar of bars) store.put(bar);
    };
  });
}

export function readSession(date) {
  return transaction("marketData", "readonly", (store, result) => {
    const request = store.index("date").getAll(date);
    request.onsuccess = () => result(request.result);
  });
}

export function sessionDates() {
  return transaction("marketData", "readonly", (store, result) => {
    const dates = [];
    result(dates);
    const request = store.index("date").openKeyCursor(null, "prevunique");
    request.onsuccess = () => {
      const cursor = request.result;
      if (cursor) {
        dates.push(cursor.key);
        cursor.continue();
      }
    };
  });
}

export function saveLog(entry) {
  return transaction("browserLogs", "readwrite", (store) => {
    store.add(entry);
    const count = store.count();
    count.onsuccess = () => {
      let excess = count.result - LOG_LIMIT;
      if (excess <= 0) return;
      const request = store.openCursor();
      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor && excess > 0) {
          cursor.delete();
          excess -= 1;
          if (excess > 0) cursor.continue();
        }
      };
    };
  });
}

export function readLogs() {
  return transaction("browserLogs", "readonly", (store, result) => {
    const request = store.index("timestamp").getAll();
    request.onsuccess = () => result(request.result.reverse());
  });
}

export function readModel() {
  return transaction("modelResults", "readonly", (store, result) => {
    const request = store.get("latest");
    request.onsuccess = () => result(request.result);
  });
}

export function saveModel(model) {
  return transaction("modelResults", "readwrite", (store) => store.put(model, "latest"));
}

export function saveBacktestRun(run) {
  return transaction("backtestRuns", "readwrite", (store) => store.put(run));
}

export function readBacktestRun(id) {
  return transaction("backtestRuns", "readonly", (store, result) => {
    const request = store.get(id);
    request.onsuccess = () => result(request.result);
  });
}

export function listBacktestRuns() {
  return transaction("backtestRuns", "readonly", (store, result) => {
    const runs = [];
    result(runs);
    const request = store.index("createdAt").openCursor(null, "prev");
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      const { equityCurve, trades, ...summary } = cursor.value;
      runs.push(summary);
      cursor.continue();
    };
  });
}

export function setBacktestPin(id, pinned) {
  return transaction("backtestRuns", "readwrite", (store) => {
    const request = store.get(id);
    request.onsuccess = () => {
      if (request.result) store.put({ ...request.result, pinned });
    };
  });
}

export function listAdxFilterPresets() {
  return transaction("adxFilterPresets", "readonly", (store, result) => {
    const request = store.getAll();
    request.onsuccess = () => result(request.result);
  });
}

export function saveAdxFilterPreset(preset) {
  return transaction("adxFilterPresets", "readwrite", (store) => { store.add(preset); });
}

export function listPaperTrades() {
  return transaction("paperTrades", "readonly", (store, result) => {
    const request = store.getAll();
    request.onsuccess = () => result(request.result);
  });
}

export function savePaperTrades(trades) {
  return transaction("paperTrades", "readwrite", (store, result) => {
    let saved = 0;
    result(saved);
    for (const trade of trades) {
      const request = store.add(trade);
      request.onsuccess = () => result(++saved);
      request.onerror = event => {
        if (request.error.name === "ConstraintError") event.preventDefault();
      };
    }
  });
}

export function updatePaperTradeMarks(marks) {
  return transaction("paperTrades", "readwrite", (store) => {
    for (const mark of marks) {
      const request = store.get([mark.ticker, mark.entryDate]);
      request.onsuccess = () => {
        const trade = request.result;
        if (trade && trade.mark.status !== "closed" && mark.asOf >= trade.mark.asOf) {
          store.put({ ...trade, mark });
        }
      };
    }
  });
}
