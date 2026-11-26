// Routes the AI analyst never calls: they stay with the human in the dashboard. Pure data, used by the
// action parity test (server/ai/test/action-parity.test.js) and the TESTING-CHECKLIST generator. The analyst's
// action enum (server/ai/actionRegistry.js) cannot name these, and actions.js refuses their patterns anyway.
export const HUMAN_ONLY = Object.freeze({
  "GET /api/live/status": "The live switch state is shown on Paper trading > Live trading.",
  "POST /api/live/toggle": "Live trading on/off is a manual switch on Paper trading > Live trading.",
  "POST /api/live/preview": "Live Schwab order preview is human-only (Paper trading > Live trading ticket).",
  "POST /api/live/options/preview": "Live Schwab options preview is human-only.",
  "POST /api/live/confirm": "Placing a live Schwab order is human-only.",
  "POST /api/live/order-status": "Live order status refresh lives on the live-trading ticket.",
  "GET /api/live/accounts": "Live account details stay on the live-trading page.",
  "GET /api/live/orders": "Live orders stay on the live-trading page.",
  "POST /api/keys/fred": "Keys are entered by you in Research & data > Data keys; the analyst never handles secrets.",
  "POST /api/keys/fred/test": "Key tests run from the Data keys card.",
  "POST /api/keys/massive": "Keys are entered by you in Research & data > Data keys.",
  "POST /api/keys/massive/test": "Key tests run from the Data keys card.",
  "POST /api/ai/provider-key": "AI provider keys are entered on the AI analyst page.",
  "POST /api/schwab/app-keys": "Schwab app keys are entered in the Schwab setup dialog.",
  "POST /api/schwab/reauth": "Schwab sign-in needs you in the browser (Click here to reauthorize).",
});

