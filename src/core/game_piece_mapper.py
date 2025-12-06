"""
FRC Game Piece Mapper - Maps generic game piece descriptions to specific FRC game elements
This module helps users ask natural questions using generic terms like "ball" or "cube"
and maps them to specific game pieces like "Algae" or "Power Cells"
"""

import re
from typing import Dict, List, Tuple, Any

class GamePieceMapper:
    def __init__(self):
        # Game piece definitions with rich descriptors
        self.game_pieces = {
            # 2025 - Reefscape
            "algae": {
                "season": "2025",
                "game": "Reefscape",
                "generic_names": ["ball", "sphere", "round object", "green ball", "orb"],
                "official_name": "Algae",
                "description": "Green spherical ball game piece from the 2025 Reefscape season",
                "physical_properties": {
                    "shape": "sphere",
                    "color": "green", 
                    "material": "rubber/foam",
                    "diameter": "7 inches",
                    "weight": "lightweight"
                },
                "pickup_locations": ["coral stations", "ground", "algae dispensers"],
                "scoring_locations": ["net", "basket", "reef"],
                "handling_methods": ["intake", "grabber", "shooter"],
                "synonyms": ["algae ball", "green sphere", "reefscape ball"]
            },
            
            "coral": {
                "season": "2025",
                "game": "Reefscape",
                "generic_names": ["block", "cube", "rectangular object", "orange block"],
                "official_name": "Coral",
                "description": "Orange rectangular coral piece from the 2025 Reefscape season",
                "physical_properties": {
                    "shape": "rectangular block",
                    "color": "orange",
                    "material": "foam/plastic",
                    "dimensions": "approximately 6x4x4 inches",
                    "weight": "lightweight"
                },
                "pickup_locations": ["coral stations", "ground", "staging areas"],
                "scoring_locations": ["reef structure", "processors"],
                "handling_methods": ["intake", "claw", "gripper"],
                "synonyms": ["coral block", "orange cube", "reefscape coral"]
            },

            # 2024 - Crescendo
            "note": {
                "season": "2024",
                "game": "Crescendo",
                "generic_names": ["ring", "donut", "circular object", "orange ring", "disc"],
                "official_name": "Note",
                "description": "Orange foam ring from the 2024 Crescendo season",
                "physical_properties": {
                    "shape": "ring/torus",
                    "color": "orange",
                    "material": "foam",
                    "outer_diameter": "14 inches",
                    "inner_diameter": "4 inches",
                    "weight": "lightweight"
                },
                "pickup_locations": ["centerline", "wing", "source", "ground"],
                "scoring_locations": ["speaker", "amp", "trap"],
                "handling_methods": ["intake", "shooter", "launcher"],
                "synonyms": ["crescendo note", "orange ring", "foam ring", "music note"]
            },

            # 2023 - Charged Up
            "cone": {
                "season": "2023", 
                "game": "Charged Up",
                "generic_names": ["cone", "yellow cone", "triangular object", "traffic cone"],
                "official_name": "Cone",
                "description": "Yellow traffic cone-shaped game piece from the 2023 Charged Up season",
                "physical_properties": {
                    "shape": "cone",
                    "color": "yellow",
                    "material": "plastic/rubber",
                    "height": "approximately 12 inches",
                    "base_diameter": "6 inches",
                    "weight": "lightweight"
                },
                "pickup_locations": ["loading zone", "ground", "single substation", "double substation"],
                "scoring_locations": ["grid", "high node", "mid node", "low node"],
                "handling_methods": ["claw", "gripper", "intake"],
                "synonyms": ["traffic cone", "yellow cone", "charged up cone"]
            },

            "cube": {
                "season": "2023",
                "game": "Charged Up", 
                "generic_names": ["cube", "purple cube", "block", "square object", "box"],
                "official_name": "Cube",
                "description": "Purple cube-shaped game piece from the 2023 Charged Up season",
                "physical_properties": {
                    "shape": "cube",
                    "color": "purple",
                    "material": "foam/plastic",
                    "dimensions": "approximately 9.5x9.5x9.5 inches",
                    "weight": "lightweight"
                },
                "pickup_locations": ["loading zone", "ground", "single substation", "double substation"],
                "scoring_locations": ["grid", "high node", "mid node", "low node"],
                "handling_methods": ["intake", "claw", "gripper"],
                "synonyms": ["purple cube", "foam cube", "charged up cube"]
            },

            # 2022 - Rapid React
            "cargo": {
                "season": "2022",
                "game": "Rapid React",
                "generic_names": ["ball", "sphere", "round object", "red ball", "blue ball"],
                "official_name": "Cargo",
                "description": "Red and blue foam balls from the 2022 Rapid React season",
                "physical_properties": {
                    "shape": "sphere",
                    "color": "red or blue (alliance specific)",
                    "material": "foam",
                    "diameter": "9.5 inches",
                    "weight": "lightweight"
                },
                "pickup_locations": ["terminal", "ground", "hangar"],
                "scoring_locations": ["upper hub", "lower hub"],
                "handling_methods": ["intake", "shooter", "conveyor"],
                "synonyms": ["cargo ball", "foam ball", "rapid react ball", "red cargo", "blue cargo"]
            },

            # 2021 - Infinite Recharge at Home / 2020
            "power_cell": {
                "season": "2020/2021",
                "game": "Infinite Recharge",
                "generic_names": ["ball", "sphere", "round object", "yellow ball", "fuel"],
                "official_name": "Power Cell",
                "description": "Yellow foam balls from the 2020/2021 Infinite Recharge season",
                "physical_properties": {
                    "shape": "sphere",
                    "color": "yellow",
                    "material": "foam",
                    "diameter": "7 inches", 
                    "weight": "lightweight"
                },
                "pickup_locations": ["loading bay", "ground", "trench"],
                "scoring_locations": ["power port", "upper goal", "lower goal"],
                "handling_methods": ["intake", "shooter", "conveyor"],
                "synonyms": ["power cell ball", "yellow ball", "infinite recharge ball", "fuel cell"]
            }
        }

        # Build reverse lookup maps
        self._build_lookup_maps()

    def _build_lookup_maps(self):
        """Build reverse lookup maps for efficient searching"""
        self.generic_to_specific = {}
        self.synonym_to_specific = {}
        
        for piece_id, piece_data in self.game_pieces.items():
            # Map generic names to specific pieces
            for generic_name in piece_data["generic_names"]:
                if generic_name not in self.generic_to_specific:
                    self.generic_to_specific[generic_name] = []
                self.generic_to_specific[generic_name].append(piece_id)
            
            # Map synonyms to specific pieces
            for synonym in piece_data["synonyms"]:
                self.synonym_to_specific[synonym.lower()] = piece_id

    def enhance_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Enhance a user query by mapping generic terms to specific game pieces
        Returns: (enhanced_query, list_of_matched_pieces)
        """
        enhanced_query = query.lower()
        matched_pieces = []
        
        # Check for direct synonym matches first
        for synonym, piece_id in self.synonym_to_specific.items():
            if synonym in enhanced_query:
                matched_pieces.append(piece_id)
        
        # Check for generic term matches
        for generic_term, piece_ids in self.generic_to_specific.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(generic_term) + r'\b'
            if re.search(pattern, enhanced_query, re.IGNORECASE):
                matched_pieces.extend(piece_ids)
                
                # Add specific game piece names to query
                for piece_id in piece_ids:
                    piece_data = self.game_pieces[piece_id]
                    official_name = piece_data["official_name"]
                    season = piece_data["season"]
                    
                    # Add context to the query
                    enhanced_query += f" {official_name} {season}"
        
        # Remove duplicates while preserving order
        matched_pieces = list(dict.fromkeys(matched_pieces))
        
        return enhanced_query, matched_pieces

    def get_context_for_pieces(self, piece_ids: List[str]) -> str:
        """
        Generate additional context for matched game pieces
        """
        if not piece_ids:
            return ""
        
        context_parts = []
        context_parts.append("GAME PIECE CONTEXT:")
        context_parts.append("=" * 50)
        
        for piece_id in piece_ids:
            if piece_id in self.game_pieces:
                piece = self.game_pieces[piece_id]
                context_parts.append(f"\n{piece['official_name']} ({piece['season']} {piece['game']}):")
                context_parts.append(f"Description: {piece['description']}")
                context_parts.append(f"Shape: {piece['physical_properties']['shape']}")
                context_parts.append(f"Color: {piece['physical_properties']['color']}")
                context_parts.append(f"Pickup locations: {', '.join(piece['pickup_locations'])}")
                context_parts.append(f"Scoring locations: {', '.join(piece['scoring_locations'])}")
                context_parts.append(f"Handling methods: {', '.join(piece['handling_methods'])}")
        
        context_parts.append("\n" + "=" * 50)
        return "\n".join(context_parts)

    def get_piece_info(self, piece_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific game piece"""
        return self.game_pieces.get(piece_id, {})

    def search_by_properties(self, **kwargs) -> List[str]:
        """
        Search for game pieces by properties
        Example: search_by_properties(color="yellow", shape="sphere")
        """
        matching_pieces = []
        
        for piece_id, piece_data in self.game_pieces.items():
            match = True
            for prop, value in kwargs.items():
                if prop in piece_data.get("physical_properties", {}):
                    if piece_data["physical_properties"][prop].lower() != value.lower():
                        match = False
                        break
                elif prop == "season":
                    if piece_data["season"] != value:
                        match = False
                        break
                elif prop == "game":
                    if piece_data["game"].lower() != value.lower():
                        match = False
                        break
            
            if match:
                matching_pieces.append(piece_id)
        
        return matching_pieces

    def get_all_seasons(self) -> List[str]:
        """Get all available seasons"""
        seasons = set()
        for piece_data in self.game_pieces.values():
            seasons.add(piece_data["season"])
        return sorted(list(seasons))

    def get_pieces_by_season(self, season: str) -> List[str]:
        """Get all game pieces for a specific season"""
        pieces = []
        for piece_id, piece_data in self.game_pieces.items():
            if piece_data["season"] == season:
                pieces.append(piece_id)
        return pieces

# Example usage and testing
if __name__ == "__main__":
    mapper = GamePieceMapper()
    
    # Test queries
    test_queries = [
        "How do I pick up a ball?",
        "What's the best way to score cubes?",
        "Can you show me cone intake mechanisms?",
        "How do I shoot rings into the speaker?",
        "What are the dimensions of the yellow sphere?",
        "How to handle orange blocks in 2025?"
    ]
    
    print("Testing Game Piece Mapper")
    print("=" * 50)
    
    for query in test_queries:
        enhanced_query, matched_pieces = mapper.enhance_query(query)
        context = mapper.get_context_for_pieces(matched_pieces)
        
        print(f"\nOriginal: {query}")
        print(f"Enhanced: {enhanced_query}")
        print(f"Matched pieces: {matched_pieces}")
        if context:
            print(f"Context:\n{context}")
        print("-" * 30)
