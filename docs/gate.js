/*
 * gate.js -- the password door in front of every page of this site.
 *
 * HOW IT WORKS. The page markup is delivered hidden (the .site-content div
 * carries the `hidden` attribute). Nothing is shown until the visitor types the
 * password. What is stored below is not the password itself but the SHA-256
 * hash of "emccd_bintool::" + password, so the word does not sit in plain sight
 * in this file. Once accepted, sessionStorage remembers the unlock for the rest
 * of the browser session, so moving between the pages of this site does not ask
 * again; closing the browser forgets it.
 *
 * WHAT THIS IS NOT. This is a doorway lock on a static site: it keeps a page out
 * of casual view and out of search results. It is not encryption, and it is not
 * a substitute for keeping anything genuinely confidential off a public web
 * server in the first place. Everything the site shows is already public science
 * and public code; the password is here so the pages stay a working document for
 * the group rather than a published one.
 */

const PASSWORD_HASH =
  '42bf461debfa6233b7076c1e927077f357a2c656ae0233775a6a032e6a496b04';
const SALT = 'emccd_bintool::';
const SESSION_KEY = 'emccd_bintool_unlocked';

const overlay = document.getElementById('passwordOverlay');
const content = document.getElementById('siteContent');
const form = document.getElementById('passwordForm');
const input = document.getElementById('passwordInput');
const error = document.getElementById('passwordError');

const unlock = () => {
  overlay.hidden = true;
  content.hidden = false;
};

const fail = (message) => {
  error.textContent = message;
  error.hidden = false;
  input.value = '';
  input.focus();
};

async function sha256(text) {
  // crypto.subtle exists in every modern browser over https (and over file://
  // in Chrome and Firefox). If it is missing, say so plainly rather than
  // failing silently or, worse, letting everyone in.
  if (!window.crypto || !window.crypto.subtle) {
    return null;
  }
  const bytes = new TextEncoder().encode(text);
  const digest = await window.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

if (sessionStorage.getItem(SESSION_KEY) === 'true') {
  unlock();
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const typed = input.value.trim();
  const hash = await sha256(SALT + typed);

  if (hash === null) {
    fail('This browser cannot check the password here. Open the site over https.');
    return;
  }
  if (hash === PASSWORD_HASH) {
    sessionStorage.setItem(SESSION_KEY, 'true');
    unlock();
    return;
  }
  fail('Wrong password.');
});
