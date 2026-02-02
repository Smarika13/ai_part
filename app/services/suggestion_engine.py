"""
Smart Suggestion Engine for Chitwan Wildlife Chatbot
Provides context-aware follow-up questions based on conversation history
"""

import re
from typing import List, Dict

class SuggestionEngine:
    def __init__(self):
        # Keyword-based suggestion mappings
        self.suggestion_rules = {
            # Wildlife-related
            'birds': [
                "Which birds are endangered in Chitwan?",
                "What's the best time for bird watching?",
                "How much does a bird watching tour cost?",
                "Where can I spot rare birds?"
            ],
            'mammals': [
                "Which mammals are endangered?",
                "What activities let me see mammals up close?",
                "Tell me about the One-horned Rhinoceros",
                "Can I see Bengal Tigers in Chitwan?"
            ],
            'reptiles': [
                "Are there crocodiles in Chitwan?",
                "Which reptiles are dangerous?",
                "Where can I safely see reptiles?",
                "Tell me about the Gharial crocodile"
            ],
            'rhino': [
                "How many rhinos are in Chitwan?",
                "What's the best way to see rhinos?",
                "Are rhinos dangerous?",
                "Conservation status of rhinos?"
            ],
            'tiger': [
                "How many tigers are in Chitwan?",
                "What are the chances of seeing a tiger?",
                "Which safari is best for tiger spotting?",
                "Tell me about Bengal Tigers"
            ],
            'elephant': [
                "Tell me about wild elephants in Chitwan",
                "What's the difference between wild and domestic elephants?",
                "How much is an elephant safari?",
                "Is elephant safari ethical?"
            ],
            
            # Activity-related
            'safari': [
                "What's the difference between jeep and elephant safari?",
                "Which safari offers the best wildlife viewing?",
                "What's included in the safari price?",
                "Can I book a private safari?"
            ],
            'jeep safari': [
                "What time does jeep safari start?",
                "How long is the jeep safari?",
                "What animals can I see on jeep safari?",
                "Price for jeep safari?"
            ],
            'elephant safari': [
                "How long is the elephant safari?",
                "Is elephant safari safe?",
                "What's the best time for elephant safari?",
                "Price for elephant safari?"
            ],
            'bird watching': [
                "What equipment do I need for bird watching?",
                "Best season for bird watching?",
                "How many bird species are in Chitwan?",
                "Price for bird watching tour?"
            ],
            'jungle walk': [
                "Is jungle walk safe?",
                "What should I bring for jungle walk?",
                "How long is the jungle walk?",
                "Price for jungle walk?"
            ],
            'canoe': [
                "How long is the canoe ride?",
                "What can I see during canoe safari?",
                "Is canoe safari safe?",
                "Price for canoe safari?"
            ],
            'tharu': [
                "What is Tharu culture?",
                "When is the Tharu cultural program?",
                "What happens in the cultural program?",
                "Can I visit a Tharu village?"
            ],
            
            # Pricing-related
            'price': [
                "What's the difference between domestic and tourist prices?",
                "Are there package deals?",
                "Which activities are most affordable?",
                "What payment methods are accepted?"
            ],
            'cost': [
                "What's included in the price?",
                "Are there group discounts?",
                "Which is the cheapest activity?",
                "Which is the most expensive activity?"
            ],
            
            # General planning
            'visit': [
                "What's the best time to visit Chitwan?",
                "How many days should I spend in Chitwan?",
                "What should I pack for Chitwan?",
                "Where should I stay in Chitwan?"
            ],
            'season': [
                "What's the best season for wildlife viewing?",
                "Is Chitwan open year-round?",
                "What's the weather like in different seasons?",
                "When is monsoon season?"
            ],
            'endangered': [
                "What conservation efforts are in place?",
                "Can I support conservation during my visit?",
                "Which species need protection most?",
                "Success stories of conservation?"
            ],
            
            # Timing-related
            'timing': [
                "What are the park opening hours?",
                "Best time of day for wildlife spotting?",
                "How early should I start activities?",
                "Evening activity options?"
            ]
        }
        
        # Default suggestions when no specific match
        self.default_suggestions = [
            "What activities are available in Chitwan?",
            "Tell me about the wildlife in Chitwan",
            "What's the best time to visit?",
            "Show me activity prices for tourists"
        ]
        
        # Category-based suggestions
        self.category_suggestions = {
            'wildlife': [
                "Tell me about endangered species",
                "What birds can I see?",
                "Information about One-horned Rhino",
                "Are there Bengal Tigers here?"
            ],
            'activities': [
                "Compare jeep safari vs elephant safari",
                "What's included in each activity?",
                "Best activities for families",
                "Adventure activities available?"
            ],
            'planning': [
                "Create a 2-day itinerary",
                "Best season to visit Chitwan",
                "Accommodation recommendations",
                "How to reach Chitwan from Kathmandu?"
            ]
        }

    def get_suggestions(self, user_query: str, bot_response: str, conversation_history: List[Dict] = None) -> List[str]:
        """
        Generate smart suggestions based on context
        
        Args:
            user_query: The user's question
            bot_response: The bot's answer
            conversation_history: Previous conversation turns
            
        Returns:
            List of 3-4 relevant follow-up questions
        """
        suggestions = []
        user_query_lower = user_query.lower()
        bot_response_lower = bot_response.lower()
        
        # 1. Check for keyword matches in user query
        for keyword, keyword_suggestions in self.suggestion_rules.items():
            if keyword in user_query_lower or keyword in bot_response_lower:
                suggestions.extend(keyword_suggestions)
        
        # 2. Remove duplicates while preserving order
        suggestions = list(dict.fromkeys(suggestions))
        
        # 3. If we have suggestions, return top 4
        if suggestions:
            return suggestions[:4]
        
        # 4. If no specific matches, use default suggestions
        return self.default_suggestions[:4]

    def get_contextual_suggestions(self, detected_entities: Dict[str, List[str]]) -> List[str]:
        """
        Get suggestions based on detected entities in the conversation
        
        Args:
            detected_entities: Dict with keys like 'animals', 'activities', 'prices'
            
        Returns:
            List of contextual suggestions
        """
        suggestions = []
        
        if 'animals' in detected_entities and detected_entities['animals']:
            animal = detected_entities['animals'][0]
            suggestions.append(f"Tell me more about {animal}")
            suggestions.append(f"Where can I see {animal}?")
            suggestions.append(f"Conservation status of {animal}?")
        
        if 'activities' in detected_entities and detected_entities['activities']:
            activity = detected_entities['activities'][0]
            suggestions.append(f"How much does {activity} cost?")
            suggestions.append(f"What's the best time for {activity}?")
            suggestions.append(f"How long is the {activity}?")
        
        return suggestions[:4]

    def get_smart_followups(self, user_query: str, bot_response: str) -> List[str]:
        """
        Advanced suggestion generation with response analysis
        
        Args:
            user_query: User's question
            bot_response: Bot's answer
            
        Returns:
            List of intelligent follow-up questions
        """
        suggestions = []
        
        # Check if response mentions multiple items (suggests comparison)
        if bot_response.lower().count('•') > 2 or bot_response.lower().count('\n') > 3:
            suggestions.append("Compare the top 3 options")
            suggestions.append("Which one would you recommend?")
        
        # Check if prices are mentioned
        if 'npr' in bot_response.lower() or 'price' in bot_response.lower() or any(char.isdigit() for char in bot_response):
            suggestions.append("Are there any discounts available?")
            suggestions.append("What's included in this price?")
        
        # Check if endangered/conservation mentioned
        if 'endangered' in bot_response.lower() or 'conservation' in bot_response.lower():
            suggestions.append("How can I support conservation efforts?")
            suggestions.append("What are the main threats to these species?")
        
        # Check if timing/schedule mentioned
        if 'morning' in bot_response.lower() or 'evening' in bot_response.lower() or 'time' in bot_response.lower():
            suggestions.append("What's the daily schedule for activities?")
            suggestions.append("Can I customize the timing?")
        
        return suggestions

    def format_suggestions(self, suggestions: List[str]) -> str:
        """
        Format suggestions for display
        
        Args:
            suggestions: List of suggestion strings
            
        Returns:
            Formatted string with emoji bullets
        """
        if not suggestions:
            return ""
        
        formatted = "\n\n💡 **You might also want to know:**\n"
        for i, suggestion in enumerate(suggestions[:4], 1):
            formatted += f"{i}. {suggestion}\n"
        
        return formatted

    def get_category_based_suggestions(self, category: str) -> List[str]:
        """
        Get suggestions based on conversation category
        
        Args:
            category: 'wildlife', 'activities', or 'planning'
            
        Returns:
            Category-specific suggestions
        """
        return self.category_suggestions.get(category, self.default_suggestions)