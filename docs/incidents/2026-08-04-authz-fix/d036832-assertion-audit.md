# `d036832` assertion-group audit

Compared with `c25d7e2`, `d036832` deleted or inverted five assertion groups:

1. **G1 — sentence-initial positional questions were not change requests.**
   Legitimately superseded for the promoted positional verbs: polite execution
   questions are now commands, while meta-questions remain blocked.
2. **G2 — `请把吃席挪到 wkxk` was not mutation authorization.** Legitimately
   superseded by the authorization/binding split: the request may authorize the
   intent, but the ASCII destination binds only when it is a server candidate.
3. **G3 — a verb-miss response did not say `不能授权修改草稿` and did explain
   `与历史、记忆或引用无关`.** Wrongly dropped. The positional incident now has a
   different block reason, but the two 2026-08-04 wording guarantees still apply
   to the surviving `verb_not_matched` branch and are pinned there again.
4. **G4 — `把吃席的编码放到 wkxk` was blocked and produced the exact self-checked
   suggestion `@我 顺延「吃席」到 wkxk`.** Wrongly inverted: direct success without
   a server candidate violated destination provenance. The blocked/suggestion
   assertion is restored with the A1 fix.
5. **G5 — replaying that exact suggestion succeeded.** Legitimately superseded as
   a duplicate: replayability survived in both the incident-phrasing corpus and
   the authorization-tool integration test. The local replay is also retained
   beside G4 so the corrected remediation path is explicit.
