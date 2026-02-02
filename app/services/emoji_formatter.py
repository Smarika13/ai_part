"""
Emoji Formatter for Chitwan Wildlife Chatbot
Adds context-aware emojis to make responses more engaging and visual
"""

import re
from typing import Dict, List

class EmojiFormatter:
    def __init__(self):
        # Wildlife emojis
        self.animal_emojis = {
            # Mammals
            'tiger': '🐯',
            'bengal tiger': '🐯',
            'royal bengal tiger': '🐯',
            'rhino': '🦏',
            'rhinoceros': '🦏',
            'one-horned rhino': '🦏',
            'one-horned rhinoceros': '🦏',
            'elephant': '🐘',
            'asian elephant': '🐘',
            'leopard': '🐆',
            'sloth bear': '🐻',
            'bear': '🐻',
            'deer': '🦌',
            'spotted deer': '🦌',
            'sambar deer': '🦌',
            'wild boar': '🐗',
            'boar': '🐗',
            'monkey': '🐒',
            'langur': '🐒',
            'rhesus macaque': '🐒',
            'jackal': '🦊',
            'fox': '🦊',
            'mongoose': '🦡',
            'otter': '🦦',
            'dolphin': '🐬',
            'gangetic dolphin': '🐬',
            
            # Birds
            'bird': '🦅',
            'eagle': '🦅',
            'vulture': '🦅',
            'peacock': '🦚',
            'peafowl': '🦚',
            'duck': '🦆',
            'goose': '🦆',
            'stork': '🦩',
            'crane': '🦩',
            'heron': '🦩',
            'egret': '🦩',
            'kingfisher': '🐦',
            'hornbill': '🦜',
            'parrot': '🦜',
            'owl': '🦉',
            'woodpecker': '🐦',
            'flycatcher': '🐦',
            'warbler': '🐦',
            'tern': '🐦',
            'ibis': '🦆',
            
            # Reptiles
            'crocodile': '🐊',
            'gharial': '🐊',
            'mugger crocodile': '🐊',
            'snake': '🐍',
            'python': '🐍',
            'cobra': '🐍',
            'lizard': '🦎',
            'monitor lizard': '🦎',
            'turtle': '🐢',
            'tortoise': '🐢',
        }
        
        # Activity emojis
        self.activity_emojis = {
            'jeep safari': '🚙',
            'safari': '🚙',
            'jeep': '🚙',
            'elephant safari': '🐘',
            'elephant back': '🐘',
            'elephant ride': '🐘',
            'bird watching': '🦅',
            'birding': '🦅',
            'jungle walk': '🚶',
            'nature walk': '🚶',
            'walking': '🚶',
            'canoe': '🛶',
            'canoe safari': '🛶',
            'boat': '🛶',
            'tharu': '🎭',
            'cultural program': '🎭',
            'culture': '🎭',
            'dance': '💃',
            'museum': '🏛️',
            'tharu museum': '🏛️',
        }
        
        # Time/Schedule emojis
        self.time_emojis = {
            'morning': '🌅',
            'sunrise': '🌅',
            'afternoon': '☀️',
            'evening': '🌆',
            'sunset': '🌇',
            'night': '🌙',
            'dawn': '🌄',
            'dusk': '🌆',
        }
        
        # Status/Conservation emojis
        self.status_emojis = {
            'endangered': '⚠️',
            'vulnerable': '⚠️',
            'threatened': '⚠️',
            'critically endangered': '🚨',
            'extinct': '❌',
            'protected': '🛡️',
            'conservation': '🌱',
            'rare': '💎',
        }
        
        # General context emojis
        self.context_emojis = {
            'price': '💰',
            'cost': '💰',
            'rupee': '💵',
            'npr': '💵',
            'money': '💵',
            'payment': '💳',
            'ticket': '🎟️',
            'booking': '📅',
            'schedule': '📅',
            'timing': '⏰',
            'time': '⏰',
            'duration': '⏱️',
            'location': '📍',
            'place': '📍',
            'habitat': '🌳',
            'forest': '🌲',
            'jungle': '🌴',
            'river': '🌊',
            'water': '💧',
            'season': '🌤️',
            'weather': '🌡️',
            'temperature': '🌡️',
            'rain': '🌧️',
            'monsoon': '🌧️',
            'winter': '❄️',
            'summer': '☀️',
            'food': '🍽️',
            'restaurant': '🍽️',
            'hotel': '🏨',
            'accommodation': '🏨',
            'stay': '🏨',
            'guide': '👨‍🏫',
            'tourist': '🧳',
            'visitor': '🧳',
            'family': '👨‍👩‍👧‍👦',
            'children': '👶',
            'kids': '👶',
            'safety': '🦺',
            'danger': '⚠️',
            'warning': '⚠️',
            'tip': '💡',
            'suggestion': '💡',
            'recommendation': '✨',
            'best': '⭐',
            'popular': '⭐',
            'famous': '⭐',
        }
        
        # Combine all emoji mappings
        self.all_emojis = {
            **self.animal_emojis,
            **self.activity_emojis,
            **self.time_emojis,
            **self.status_emojis,
            **self.context_emojis
        }
        
        # Number emojis for lists
        self.number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']

    def format_response(self, text: str) -> str:
        """
        Add emojis to response text based on context
        
        Args:
            text: Original response text
            
        Returns:
            Formatted text with emojis
        """
        formatted_text = text
        
        # 1. Add emojis to specific keywords (case-insensitive)
        for keyword, emoji in self.all_emojis.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            # Only add emoji if not already present
            replacement = f"{emoji} {keyword}"
            formatted_text = re.sub(
                pattern, 
                lambda m: replacement if emoji not in formatted_text[max(0, m.start()-2):m.start()] else m.group(0),
                formatted_text, 
                flags=re.IGNORECASE,
                count=1  # Only format first occurrence to avoid emoji spam
            )
        
        # 2. Format prices with currency emoji
        formatted_text = self.format_prices(formatted_text)
        
        # 3. Format lists with number emojis
        formatted_text = self.format_lists(formatted_text)
        
        # 4. Add section headers with emojis
        formatted_text = self.format_headers(formatted_text)
        
        return formatted_text

    def format_prices(self, text: str) -> str:
        """Add currency emoji to prices"""
        # Match patterns like "NPR 500", "Rs. 500", "500 rupees"
        patterns = [
            (r'\bNPR\s+(\d+(?:,\d{3})*)', r'💰 NPR \1'),
            (r'\bRs\.?\s+(\d+(?:,\d{3})*)', r'💰 Rs. \1'),
            (r'(\d+(?:,\d{3})*)\s+rupees?', r'💰 \1 rupees'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def format_lists(self, text: str) -> str:
        """Format numbered lists with emoji numbers"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Check if line starts with a number followed by period or parenthesis
            match = re.match(r'^(\d+)[.)]\s+(.+)$', line.strip())
            if match:
                num = int(match.group(1))
                content = match.group(2)
                if num <= 10:
                    # Use emoji number
                    formatted_lines.append(f"{self.number_emojis[num-1]} {content}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def format_headers(self, text: str) -> str:
        """Add emojis to section headers"""
        header_emojis = {
            'activities': '🎯',
            'wildlife': '🦁',
            'birds': '🦅',
            'mammals': '🐾',
            'reptiles': '🦎',
            'prices': '💰',
            'pricing': '💰',
            'schedule': '📅',
            'timing': '⏰',
            'location': '📍',
            'conservation': '🌱',
            'habitat': '🌳',
            'description': '📝',
            'information': 'ℹ️',
            'tips': '💡',
            'recommendations': '✨',
        }
        
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Check if line looks like a header (all caps, ends with colon, etc.)
            if line.strip().endswith(':') and len(line.strip()) < 50:
                for keyword, emoji in header_emojis.items():
                    if keyword in line.lower():
                        if emoji not in line:
                            line = f"{emoji} {line}"
                        break
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def format_species_info(self, species_name: str, info: Dict) -> str:
        """
        Format species information with emojis
        
        Args:
            species_name: Name of the species
            info: Dictionary with species information
            
        Returns:
            Formatted string with emojis
        """
        emoji = self.get_emoji_for_species(species_name)
        
        formatted = f"\n{emoji} **{species_name}**\n"
        formatted += "─" * 40 + "\n"
        
        # Add information with appropriate emojis
        if 'nepali_name' in info:
            formatted += f"🇳🇵 Nepali Name: {info['nepali_name']}\n"
        
        if 'scientific_name' in info:
            formatted += f"🔬 Scientific Name: *{info['scientific_name']}*\n"
        
        if 'conservation_status' in info:
            status_emoji = self.status_emojis.get(info['conservation_status'].lower(), '📊')
            formatted += f"{status_emoji} Status: {info['conservation_status']}\n"
        
        if 'habitat' in info:
            formatted += f"🌳 Habitat: {info['habitat']}\n"
        
        if 'description' in info:
            formatted += f"📝 Description: {info['description']}\n"
        
        return formatted

    def get_emoji_for_species(self, species_name: str) -> str:
        """Get the most appropriate emoji for a species"""
        species_lower = species_name.lower()
        
        # Check for exact or partial matches
        for keyword, emoji in self.animal_emojis.items():
            if keyword in species_lower:
                return emoji
        
        # Default emojis by category
        if any(word in species_lower for word in ['bird', 'eagle', 'duck', 'crane', 'heron']):
            return '🦅'
        elif any(word in species_lower for word in ['snake', 'python', 'cobra']):
            return '🐍'
        elif any(word in species_lower for word in ['crocodile', 'gharial']):
            return '🐊'
        else:
            return '🐾'  # Generic wildlife emoji

    def add_visual_separators(self, text: str) -> str:
        """Add visual separators to make content more readable"""
        # Add separator before major sections
        sections = ['Activities:', 'Prices:', 'Schedule:', 'Wildlife:', 'Information:']
        
        for section in sections:
            if section in text:
                text = text.replace(section, f"\n{'─' * 40}\n{section}")
        
        return text

    def format_activity_info(self, activity: Dict) -> str:
        """
        Format activity information with emojis
        
        Args:
            activity: Dictionary with activity information
            
        Returns:
            Formatted string with emojis
        """
        name = activity.get('activity', 'Unknown Activity')
        emoji = self.activity_emojis.get(name.lower(), '🎯')
        
        formatted = f"\n{emoji} **{name}**\n"
        formatted += "─" * 40 + "\n"
        
        if 'prices' in activity:
            formatted += "💰 **Prices:**\n"
            prices = activity['prices']
            if 'domestic' in prices:
                formatted += f"  🇳🇵 Domestic: NPR {prices['domestic']}\n"
            if 'SAARC' in prices:
                formatted += f"  🌏 SAARC: NPR {prices['SAARC']}\n"
            if 'tourist' in prices:
                formatted += f"  🌍 Foreign Tourist: NPR {prices['tourist']}\n"
        
        if 'schedule' in activity:
            formatted += f"📅 Schedule: {activity['schedule']}\n"
        
        if 'timing' in activity:
            formatted += f"⏰ Timing: {activity['timing']}\n"
        
        return formatted