"""
Groups reporters by jurisdiction using ReportersMetadata.json.

Reporters belonging to exactly one jurisdiction are filed under that jurisdiction's
name. Reporters spanning multiple jurisdictions (e.g. A.2d) are placed under "Other".
"""

import unittest
from typing import Any, Dict, List, Set

def sort_reporters_to_jurisdiction(reporters_metadata: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Given a list of reporter metadata entries, returns a mapping from jurisdiction
    name to the set of reporter slugs for that jurisdiction.

    Reporters with exactly one jurisdiction are filed under that jurisdiction's 'name'.
    Reporters with zero or multiple jurisdictions are filed under 'Other'.
    Reporters missing a slug are skipped.

    Args:
        reporters_metadata: List of reporter dicts, each with 'slug' and 'jurisdictions' fields.

    Returns:
        Dict mapping jurisdiction name -> set of reporter slugs.
    """
    jurisdictions: Dict[str, Set[str]] = {"Other": set()}

    for reporter in reporters_metadata:
        slug = reporter.get("slug")
        jurs = reporter.get("jurisdictions", [])

        if not slug:
            raise ValueError(f"Reporter entry missing slug: {reporter}") 

        if len(jurs) == 1 and isinstance(jurs[0], dict) and "name" in jurs[0]:
            jur_name = jurs[0]["name"]
            if jur_name not in jurisdictions:
                jurisdictions[jur_name] = set()
            jurisdictions[jur_name].add(slug)
        else:
            jurisdictions["Other"].add(slug)

    return jurisdictions


THOMP_COOK = {
    "id": 134,
    "full_name": "Thompson & Cook's Supreme Court Reports",
    "short_name": "Thomp. & Cook",
    "start_year": 1873,
    "end_year": 1875,
    "jurisdictions": [
      {
        "id": 1,
        "name": "N.Y.",
        "name_long": "New York"
      }
    ],
    "harvard_hollis_id": [],
    "slug": "thomp-cook"
}

A3D = {
    "id": 964,
    "full_name": "West's Atlantic Reporter, Third Series",
    "short_name": "A.3d",
    "start_year": 2004,
    "end_year": 2019,
    "jurisdictions": [
      {
        "id": 56,
        "name": "D.C.",
        "name_long": "District of Columbia"
      },
      {
        "id": 8,
        "name": "Del.",
        "name_long": "Delaware"
      },
      {
        "id": 42,
        "name": "Me.",
        "name_long": "Maine"
      },
      {
        "id": 6,
        "name": "Pa.",
        "name_long": "Pennsylvania"
      },
      {
        "id": 9,
        "name": "Regional",
        "name_long": "Regional"
      },
      {
        "id": 15,
        "name": "R.I.",
        "name_long": "Rhode Island"
      },
      {
        "id": 39,
        "name": "U.S.",
        "name_long": "United States"
      }
    ],
    "harvard_hollis_id": [
      "001603965"
    ],
    "slug": "a3d"
}

NO_JURISDICTION = {
    "id": 999,
    "full_name": "Hypothetical Reports",
    "short_name": "Hyp.",
    "start_year": 1900,
    "end_year": 1901,
    "jurisdictions": [],
    "harvard_hollis_id": [],
    "slug": "hyp",
}


class TestSortReportersToJurisdiction(unittest.TestCase):

    def test_single_jurisdiction(self):
        # Thompson & Cook belongs only to New York
        result = sort_reporters_to_jurisdiction([THOMP_COOK])
        self.assertEqual(result["N.Y."], {"thomp-cook"})
        self.assertEqual(result["Other"], set())

    def test_multiple_jurisdictions_goes_to_other(self):
        # A.3d spans 7 jurisdictions and should land in Other, not under any of them
        result = sort_reporters_to_jurisdiction([A3D])
        self.assertIn("a3d", result["Other"])
        for jur in ("D.C.", "Del.", "Me.", "Pa.", "Regional", "R.I.", "U.S."):
            self.assertNotIn(jur, result)

    def test_zero_jurisdictions_goes_to_other(self):
        # a reporter with an empty jurisdictions list also goes to "Other" since there's no single jurisdiction to file it under
        result = sort_reporters_to_jurisdiction([NO_JURISDICTION])
        self.assertIn("hyp", result["Other"])

    def test_missing_slug_raises(self):
        # a reporter entry without a slug raises ValueError, since there's no way to identify it
        reporter = {**THOMP_COOK}
        del reporter["slug"]
        with self.assertRaises(ValueError):
            sort_reporters_to_jurisdiction([reporter])

    def test_multiple_reporters_same_jurisdiction(self):
        # two reporters both belonging to N.Y. should both appear in result["N.Y."]
        second = {**THOMP_COOK, "slug": "thomp-cook-2"}
        result = sort_reporters_to_jurisdiction([THOMP_COOK, second])
        self.assertEqual(result["N.Y."], {"thomp-cook", "thomp-cook-2"})

    def test_mixed_input(self):
        # all three cases together: single-jurisdiction reporter goes to its jurisdiction, multi-jurisdiction and zero-jurisdiction reporters both go to "Other"
        result = sort_reporters_to_jurisdiction([THOMP_COOK, A3D, NO_JURISDICTION])
        self.assertEqual(result["N.Y."], {"thomp-cook"})
        self.assertEqual(result["Other"], {"a3d", "hyp"})

    def test_empty_input(self):
        # an empty list returns {"Other": set()} with no other keys
        result = sort_reporters_to_jurisdiction([])
        self.assertEqual(result, {"Other": set()})


if __name__ == "__main__":
    unittest.main()