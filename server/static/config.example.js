/* Copy this file to `config.js` and fill in `endpoint`.
   `config.js` is gitignored, because the /exec URL is a capability URL:
   anyone who has it can write to your sheet.

   Both signup.html and dashboard.html load `config.js` and work without it -
   they just say plainly that they are not linked to a sheet yet. */
window.CUE_SIGNUP = {
  /* The Apps Script web app URL, ending in /exec. From apps-script/Code.gs:
     Deploy > New deployment > Web app > Execute as: Me > Who has access: Anyone */
  endpoint: '',

  /* Written into every row, so one sheet can hold more than one event. */
  eventId: 'hackmit-demo',

  /* Optional. The dashboard learns this from the endpoint anyway; setting it
     here just makes "Open the Google Sheet" work before the first fetch. */
  sheetUrl: '',

  /* Photo size sent from the phone. 640px of JPEG is ~40-70 KB, which keeps a
     sign-up under the Apps Script POST limit with room to spare. */
  maxEdge: 640,
  quality: 0.82,

  /* How often the dashboard re-reads the sheet. Auto-refresh is on by default,
     floored at 5000 ms. */
  pollMs: 5000,

  /* How often the LIVE DESK (index.html) re-reads the sheet, so a guest who has
     just signed up appears on the desk roster without anyone copying anything.
     A `roster` message from your backend overrides this and stops the polling.
     Floored at 2000 ms. */
  deskPollMs: 4000,
};
