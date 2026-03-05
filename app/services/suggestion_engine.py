"""
suggestion_engine.py — Smart Suggestion Engine
================================================
Fixes vs original:
  - Never repeats the question the user just asked
  - Rotates through suggestion pools so the same 3 never repeat consecutively
  - Conservation queries get varied follow-ups (not always the same 3)
  - Tracks last shown suggestions to avoid repetition
"""

import re
from typing import List, Dict


class SuggestionEngine:
    def __init__(self):
        self._last_suggestions: List[str] = []

        # Each key maps to a POOL of suggestions — we rotate through them
        self.suggestion_pools: Dict[str, List[str]] = {

            # ── Wildlife ─────────────────────────────────────────────────────
            "tiger": [
                "How many tigers are in Chitwan?",
                "What are the chances of seeing a tiger?",
                "Which safari is best for tiger spotting?",
                "Tell me about Bengal Tigers",
                "When are tigers most active?",
                "Where do tigers live in Chitwan?",
            ],
            "rhino": [
                "How many rhinos are in Chitwan?",
                "What is the best way to see rhinos?",
                "Are rhinos dangerous to tourists?",
                "Conservation status of one-horned rhinos?",
                "Where can I spot rhinos in Chitwan?",
            ],
            "elephant": [
                "Tell me about wild elephants in Chitwan",
                "What is the difference between wild and domestic elephants?",
                "How much is an elephant safari?",
                "Is elephant safari ethical?",
                "Where do elephants roam in Chitwan?",
            ],
            "bird": [
                "Which birds are endangered in Chitwan?",
                "What is the best time for bird watching?",
                "How much does a bird watching tour cost?",
                "Where can I spot rare birds?",
                "How many bird species are in Chitwan?",
                "What equipment do I need for bird watching?",
            ],
            "birds": [
                "Which birds are endangered in Chitwan?",
                "What is the best time for bird watching?",
                "How much does a bird watching tour cost?",
                "Where can I spot rare birds?",
                "What are the rarest birds in Chitwan?",
                "Best season for bird watching in Chitwan?",
            ],
            "crocodile": [
                "Are there crocodiles in Chitwan?",
                "Where can I safely see gharials?",
                "What is the difference between gharial and mugger?",
                "Conservation status of gharial crocodile?",
            ],
            "reptile": [
                "Are there crocodiles in Chitwan?",
                "Which reptiles are dangerous?",
                "Where can I safely see reptiles?",
                "Tell me about the Gharial crocodile",
            ],
            "mammal": [
                "Which mammals are endangered?",
                "What activities let me see mammals up close?",
                "Tell me about the One-horned Rhinoceros",
                "Can I see Bengal Tigers in Chitwan?",
            ],

            # ── Conservation ─────────────────────────────────────────────────
            "endangered": [
                "What conservation efforts are in place?",
                "Can I support conservation during my visit?",
                "Which species need protection most?",
                "Success stories of conservation in Chitwan?",
                "How has the rhino population changed over time?",
                "What threats do tigers face in Chitwan?",
                "Which birds are critically endangered here?",
                "What is being done to protect gharials?",
            ],
            "conservation": [
                "Which species are critically endangered?",
                "How can tourists help conservation?",
                "What anti-poaching measures exist?",
                "Has wildlife population improved recently?",
                "Which animals have recovered thanks to conservation?",
                "What is the biggest threat to Chitwan wildlife?",
            ],
            "vulnerable": [
                "What does vulnerable status mean?",
                "Which vulnerable species can I see?",
                "How are vulnerable species being protected?",
                "What threats do vulnerable animals face?",
            ],
            "critically": [
                "How many critically endangered species are in Chitwan?",
                "What can tourists do to help?",
                "Which critically endangered birds are here?",
                "Conservation success stories in Nepal?",
            ],

            # ── Activities ───────────────────────────────────────────────────
            "safari": [
                "What is the difference between jeep and elephant safari?",
                "Which safari offers the best wildlife viewing?",
                "What is included in the safari price?",
                "Can I book a private safari?",
                "What time does jeep safari start?",
            ],
            "jeep": [
                "What time does jeep safari start?",
                "How long is the jeep safari?",
                "What animals can I see on jeep safari?",
                "Price for jeep safari?",
            ],
            "canoe": [
                "How long is the canoe ride?",
                "What can I see during canoe safari?",
                "Is canoe safari safe?",
                "Price for canoe safari?",
            ],
            "jungle walk": [
                "Is jungle walk safe?",
                "What should I bring for jungle walk?",
                "How long is the jungle walk?",
                "Price for jungle walk?",
            ],
            "tharu": [
                "What is Tharu culture?",
                "When is the Tharu cultural program?",
                "What happens in the cultural program?",
                "Can I visit a Tharu village?",
            ],

            # ── Pricing ───────────────────────────────────────────────────────
            "price": [
                "What is the difference between domestic and tourist prices?",
                "Are there package deals available?",
                "Which activities are most affordable?",
                "What payment methods are accepted?",
                "Which activity gives the best value?",
            ],
            "cost": [
                "What is included in the price?",
                "Are there group discounts?",
                "Which is the cheapest activity?",
                "Which is the most expensive activity?",
            ],

            # ── Planning ──────────────────────────────────────────────────────
            "visit": [
                "What is the best time to visit Chitwan?",
                "How many days should I spend in Chitwan?",
                "What should I pack for Chitwan?",
                "Where should I stay in Chitwan?",
            ],
            "season": [
                "What is the best season for wildlife viewing?",
                "Is Chitwan open year-round?",
                "What is the weather like in different seasons?",
                "When is monsoon season in Chitwan?",
            ],
        }

        self.default_suggestions = [
            "What activities are available in Chitwan?",
            "Tell me about the wildlife in Chitwan",
            "What is the best time to visit?",
            "How much does a jeep safari cost?",
            "Which birds are endangered in Chitwan?",
            "How many tigers are in Chitwan?",
        ]

        # ── Nepali suggestion pools (mirrors English pools) ───────────────────
        self.nepali_suggestion_pools: Dict[str, List[str]] = {
            "tiger": [
                "चितवनमा कति बाघ छन्?",
                "बाघ देख्ने सम्भावना कति छ?",
                "बाघ हेर्न कुन सफारी राम्रो छ?",
                "बंगाल बाघको बारेमा बताउनुस्",
                "बाघ कहिले सक्रिय हुन्छन्?",
            ],
            "rhino": [
                "चितवनमा कति गैंडा छन्?",
                "गैंडा हेर्ने राम्रो तरिका के हो?",
                "के गैंडा पर्यटकका लागि खतरनाक छन्?",
                "गैंडा चितवनमा कहाँ भेटिन्छन्?",
            ],
            "elephant": [
                "चितवनका जंगली हात्तीको बारेमा बताउनुस्",
                "हात्ती सफारीको मूल्य कति छ?",
                "हात्ती चितवनमा कहाँ पाइन्छन्?",
            ],
            "bird": [
                "चितवनमा कुन चराहरू लोपोन्मुख छन्?",
                "चरा हेर्न सबैभन्दा राम्रो समय कुन हो?",
                "बर्ड वाचिङ टुरको मूल्य कति छ?",
                "चितवनमा कति प्रजातिका चराहरू छन्?",
            ],
            "birds": [
                "चितवनमा कुन चराहरू लोपोन्मुख छन्?",
                "चरा हेर्ने सबैभन्दा राम्रो मौसम कुन हो?",
                "चितवनका सबभन्दा दुर्लभ चराहरू कुन हुन्?",
            ],
            "crocodile": [
                "चितवनमा घडियाल कहाँ देख्न सकिन्छ?",
                "घडियाल र मगरमच्छमा के फरक छ?",
                "घडियालको संरक्षण अवस्था के हो?",
            ],
            "endangered": [
                "संरक्षणका लागि के-के प्रयासहरू छन्?",
                "भ्रमणका बेला संरक्षणमा कसरी सहयोग गर्ने?",
                "कुन प्रजातिलाई सबभन्दा बढी संरक्षण चाहिन्छ?",
                "चितवनमा गैंडाको संख्या कसरी बढ्यो?",
            ],
            "conservation": [
                "कुन प्रजातिहरू अति संकटापन्न छन्?",
                "पर्यटकले संरक्षणमा कसरी मद्दत गर्न सक्छन्?",
                "चितवनमा वन्यजन्तुको संख्या बढेको छ?",
            ],
            "safari": [
                "जीप र हात्ती सफारीमा के फरक छ?",
                "कुन सफारीमा सबभन्दा बढी वन्यजन्तु देखिन्छ?",
                "जीप सफारी कहिले सुरु हुन्छ?",
                "सफारीमा के-के समावेश हुन्छ?",
            ],
            "jeep": [
                "जीप सफारी कहिले सुरु हुन्छ?",
                "जीप सफारी कति समयको हुन्छ?",
                "जीप सफारीमा कुन जनावरहरू देखिन्छन्?",
                "जीप सफारीको मूल्य कति हो?",
            ],
            "canoe": [
                "डुङ्गा सफारी कति लामो छ?",
                "डुङ्गा सफारीमा के देख्न सकिन्छ?",
                "के डुङ्गा सफारी सुरक्षित छ?",
                "डुङ्गा सफारीको मूल्य कति छ?",
            ],
            "jungle walk": [
                "जंगल हिँडाइ सुरक्षित छ?",
                "जंगल हिँडाइको लागि के-के लैजाने?",
                "जंगल हिँडाइको मूल्य कति छ?",
            ],
            "price": [
                "कुन गतिविधि सबभन्दा सस्तो छ?",
                "नेपाली र विदेशी टिकटमा के फरक छ?",
                "कुन गतिविधि सबभन्दा राम्रो मूल्यमा छ?",
            ],
            "cost": [
                "कुन गतिविधि सबभन्दा सस्तो छ?",
                "समूहका लागि छुट पाइन्छ?",
                "कुन गतिविधि सबभन्दा महंगो छ?",
            ],
            "visit": [
                "चितवन भ्रमणको सबैभन्दा राम्रो समय कुन हो?",
                "चितवनमा कति दिन बिताउनु पर्छ?",
                "चितवन जाँदा के-के सामान लैजाने?",
                "चितवनमा कहाँ बस्ने?",
            ],
            "season": [
                "वन्यजन्तु हेर्नका लागि कुन मौसम राम्रो छ?",
                "के चितवन वर्षभरि खुला रहन्छ?",
                "मनसुनको समयमा चितवन कस्तो हुन्छ?",
            ],
        }

        self.nepali_default_suggestions = [
            "चितवनमा कुन-कुन गतिविधिहरू छन्?",
            "चितवनका वन्यजन्तुको बारेमा बताउनुस्",
            "चितवन भ्रमणको राम्रो समय कुन हो?",
            "जीप सफारीको मूल्य कति छ?",
            "चितवनमा कति बाघ छन्?",
            "कुन चराहरू लोपोन्मुख छन्?",
        ]

    def get_raw_suggestions(self, user_query: str, bot_response: str,
                            match_query: str = None, language: str = "en") -> List[str]:
        """
        Return 3 non-repeating, context-aware suggestions.
        - match_query: English translation of query for keyword matching (use when user wrote Nepali)
        - language: 'en' or 'ne' — determines which suggestion pool to draw from
        """
        # Use translated English query for keyword matching if provided
        q_for_match = (match_query or user_query).lower()
        r_lower     = bot_response.lower()

        # Pick the right pool based on language
        pools    = self.nepali_suggestion_pools if language == "ne" else self.suggestion_pools
        defaults = self.nepali_default_suggestions if language == "ne" else self.default_suggestions

        # Build candidate pool from all matching keywords
        pool: List[str] = []
        for keyword, suggestions in pools.items():
            if keyword in q_for_match or keyword in r_lower:
                pool.extend(suggestions)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for s in pool:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        pool = deduped

        # Fall back to language-appropriate defaults if no keyword matched
        if not pool:
            pool = defaults[:]

        # Filter out the user's own question
        user_q_clean = re.sub(r"[?!.,]", "", user_query.strip().lower())
        def not_same_as_query(s: str) -> bool:
            return re.sub(r"[?!.,]", "", s.strip().lower()) != user_q_clean

        # Priority 1: fresh suggestions (not seen last turn, not same as query)
        fresh = [s for s in pool if not_same_as_query(s) and s not in self._last_suggestions]

        # Priority 2: if not enough fresh, rotate from pool (skip query only)
        if len(fresh) < 3:
            pool_filtered = [s for s in pool if not_same_as_query(s)]
            seen_in_pool  = [s for s in pool_filtered if s in self._last_suggestions]
            fresh = pool_filtered[:3] if len(pool_filtered) >= 3 else (pool_filtered + seen_in_pool)

        # Priority 3: pad with language defaults if still not enough
        if len(fresh) < 3:
            extras = [s for s in defaults if not_same_as_query(s) and s not in fresh]
            fresh = (fresh + extras)[:3]

        result = fresh[:3]

        # Pad if somehow fewer than 3
        if len(result) < 3:
            extras = [s for s in defaults if s not in result]
            result = (result + extras)[:3]

        # Remember what we showed to avoid repeating next turn
        self._last_suggestions = result[:]

        return result