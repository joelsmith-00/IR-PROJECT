"""Unit tests for irdiv/real.py::_title. The judgments carry no party-name caption, so a title is the cleaned docket line.
Inputs below are real header shapes taken from data/real/cases_raw.json (including the ones with their first characters cut off).

    python -m unittest discover -s tests -p "test_real_titles.py" -v        # from the repo root
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
from irdiv.real import _title, load_real

FB = "FALLBACK"
RAW = os.path.join(HERE, "..", "data", "real", "cases_raw.json")


class TitleParsing(unittest.TestCase):
    def t(self, text):
        return _title(text, FB)

    def test_plain_docket_line(self):
        self.assertEqual(self.t("Civil Appeal No. 50 of 1961.\nAppeal by special leave from the Award dated March 10, 1959."), "Civil Appeal No. 50 of 1961")
        self.assertEqual(self.t("Appeal No. 1695 of 1993.\nFrom the Judgment and Order dated 5.4.1991 of the Bombay High Court."), "Appeal No. 1695 of 1993")

    def test_truncated_type_word_is_repaired_when_unambiguous(self):
        self.assertEqual(self.t("ivil Appeal No. 325 of 1965.\nAppeal by special leave from the Award."), "Civil Appeal No. 325 of 1965")
        self.assertEqual(self.t("il Appeal Nos. 931 and 1149 of 1965.\nAppeals from the judgment and decree."), "Civil Appeal Nos. 931 and 1149 of 1965")
        self.assertEqual(self.t("iminal Appeals Nos.\n160 to 162 of 1960.\nAppeals by special leave."), "Criminal Appeals Nos. 160 to 162 of 1960")
        self.assertEqual(self.t("etition (Civil) Nos.\n12676 77 of 1985 etc.\n(Under Article 32 of the Constitution of India)."), "Petition (Civil) Nos. 12676 77 of 1985")

    def test_ambiguous_fragment_is_dropped_not_guessed(self):
        """'l' could be the end of Civil or Criminal: say neither."""
        self.assertEqual(self.t("l Appeals Nos.\n15 to 19 of 1962.\nAppeal from the Judgment and order."), "Appeals Nos. 15 to 19 of 1962")

    def test_jurisdiction_noise_prefix_is_removed(self):
        for pre in ("ION: ", "N: ", ": ", "ON: "):
            self.assertEqual(self.t(f"{pre}Criminal Appeal No. 87 of 1967.\n721 Appeal by special leave from the judgment."), "Criminal Appeal No. 87 of 1967", pre)

    def test_page_noise_after_the_year_is_dropped(self):
        self.assertEqual(self.t("Civil Appeal No. 378 of 1987.\n237 From the Judgment and Order dated 4.8.1986."), "Civil Appeal No. 378 of 1987")
        self.assertEqual(self.t("cellaneous Petition No. 32937 of 1988.\n345 IN Writ Petition No. 12 183 of 1985."), "Miscellaneous Petition No. 32937 of 1988")

    def test_small_normalisations(self):
        self.assertEqual(self.t("iminal Appeal No.112 of 1965.\nAppeal by special leave."), "Criminal Appeal No. 112 of 1965")
        self.assertEqual(self.t("ivil Appeal No 1514 of 1979.\nFrom the Judgment and Order."), "Civil Appeal No. 1514 of 1979")
        self.assertEqual(self.t("ivil Appeal No. 196 of 1 970.\nFrom the judgment and decree."), "Civil Appeal No. 196 of 1970")
        self.assertEqual(self.t("Appeals, Nos. 275 276 of 1963.\nAppeals by special leave."), "Appeals Nos. 275 276 of 1963")
        self.assertEqual(self.t("ivil appeal No. 2197 From the Judgment and order dated 16.4.70."), "Civil Appeal No. 2197")

    def test_typeless_dockets_get_a_neutral_label(self):
        self.assertEqual(self.t("No. 65 of 1969.\nPetition under article 32 of the Constitution."), "Case No. 65 of 1969")
        self.assertEqual(self.t("o. 196 of 1954.\nUnder article 32 of the Constitution of India."), "Case No. 196 of 1954")
        self.assertEqual(self.t("(C) No. 677 of 1988.\n(Under Article 32 of the Constitution of India)."), "Petition (C) No. 677 of 1988")

    def test_text_without_a_docket_line_uses_the_fallback(self):
        self.assertEqual(self.t("it case for initiating any procee d ings for contempt against the respondents.\n[109F] & ORIGINAL JURISDICTION: Writ Petition No. 1 of 1988."), FB)
        self.assertEqual(self.t(""), FB)

    def test_cited_precedents_are_never_used_as_the_title(self):
        """Regression: the old regex returned 'Court in the State of Assam v. Ranga' for a text that merely cites that case."""
        txt = "ases Nos. 297 and 298 of 1951.\nAppeals under article 132 (1). Relying on the decision of this Court in the State of Assam vs Ranga Mahammad and others(1)."
        title = self.t(txt)
        self.assertEqual(title, "Cases Nos. 297 and 298 of 1951")
        self.assertNotIn(" v", title.lower().replace("civil", ""))

    def test_long_docket_lists_are_cut_at_a_word_boundary(self):
        txt = "etition Nos.\n" + " ".join(f"{i} {i + 5}," for i in range(1000, 1300, 10)) + "\nUnder Article 32."
        title = self.t(txt)
        self.assertLessEqual(len(title), 70)
        self.assertTrue(title.endswith("etc."))
        self.assertTrue(title.startswith("Petition Nos."))


@unittest.skipUnless(os.path.exists(RAW), "data/real/cases_raw.json not present")
class TitlesOnTheRealCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_real(RAW)

    def test_titles_are_unique_clean_and_short(self):
        titles = [c["title"] for c in self.cases]
        self.assertEqual(len(titles), len(set(titles)), "titles must identify a case")
        for t in titles:
            self.assertTrue(t[0].isupper() or t[0] == "(", t)       # no leftover cut-off fragments like 'ivil' or 'ION:'
            self.assertNotIn("\n", t)
            self.assertLessEqual(len(t), 70, t)

    def test_almost_every_case_gets_a_real_docket_title(self):
        fallbacks = [c["title"] for c in self.cases if c["title"].startswith("Supreme Court judgment")]
        self.assertLessEqual(len(fallbacks), 3, fallbacks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
