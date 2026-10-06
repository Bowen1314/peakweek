"""One short "look for" line per species, keyed by iNaturalist taxon id.

Leaf shape plus typical fall color, so a person can tell the trees apart on a walk.
Unknown taxon ids return None (the card then leaves the line out).
"""

from __future__ import annotations

from typing import Optional

LOOK_FOR = {
    48098: "3-5 lobes with toothed edges and V-shaped notches; scarlet, orange "
           "or yellow, often among the first to turn.",                       # red maple
    52543: "5 lobes with smooth, U-shaped notches and few teeth; glowing yellow, "
           "orange or orange-red.",                                           # sugar maple
    49658: "Star-shaped leaves with 5-7 pointed lobes; often purple, red and "
           "yellow on the same tree, spiky seed balls underneath.",           # sweetgum
    49005: "7-11 pointed, bristle-tipped lobes; russet to deep red, turns late "
           "and often holds its leaves.",                                     # northern red oak
    49202: "Oval leaves with straight parallel veins, each ending in a tooth, and "
           "smooth gray bark; golden bronze, turns late.",                    # American beech
    54802: "Glossy oval leaves with smooth edges; one of the first trees to turn, "
           "a deep scarlet.",                                                 # black gum
    54795: "Mitten-shaped, three-lobed and unlobed leaves on the same tree; "
           "orange, red, yellow or purple.",                                  # sassafras
    54763: "Broad 5-lobed leaves with pointed tips, milky sap in a broken leaf "
           "stem; turns yellow, very late.",                                  # Norway maple
}


def look_for(taxon_id) -> Optional[str]:
    """The "look for" line for a taxon id, or None if we have no note for it."""
    try:
        return LOOK_FOR.get(int(taxon_id))
    except (TypeError, ValueError):
        return None
