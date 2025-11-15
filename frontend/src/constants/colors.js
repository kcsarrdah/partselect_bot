export const colors = {
  // Primary Colors
  primary: {
    teal: '#4A7C7E',           // Navigation bar (slightly adjusted)
    tealAlt: '#3D7375',        // Search button, darker teal ✅
    gold: '#E6B951',           // Hero banner (more muted than #F4C430)
    goldAlt: '#F4C430',        // Brighter gold for accents
    lightCyan: '#F8F8F8',      // Background (almost white, not cyan)
    orange: '#F47920',         // Logo accent (NEW!)
  },

  // Secondary Colors
  secondary: {
    darkNavy: '#001F3F',       // Keep for dark text
    white: '#FFFFFF',
    lightGray: '#F5F5F5',      // Subtle backgrounds
    lightGrayAlt: '#E8E8E8',
    cream: '#FAF7F2',          // Warmer background option (NEW!)
  },

  // Accent Colors
  accent: {
    brightRed: '#E63946',
    brightRedAlt: '#D32F2F',
    darkTeal: '#2D7175',       // Icons, appliance illustrations
    magenta: '#E91E63',
    orange: '#F47920',         // Orange from logo
    orangeAlt: '#FFA500',
    green: '#28A745',          // For success states (NEW!)
  },

  // Neutral Colors
  neutral: {
    black: '#000000',
    darkGray: '#333333',       // Body text (slightly lighter)
    mediumGray: '#666666',     // Secondary text
    borderGray: '#D4D4D4',     // Borders (slightly darker)
  },
};

// Updated theme colors
export const themeColors = {
  // Navigation & Headers
  navigation: colors.primary.teal,
  header: colors.primary.teal,
  
  // Buttons & CTAs
  buttonPrimary: colors.primary.tealAlt,      // CHANGED: Use darker teal like "SEARCH"
  buttonPrimaryHover: colors.accent.darkTeal,
  buttonSecondary: colors.primary.gold,       // Gold for secondary actions
  
  // Links
  link: colors.primary.teal,
  linkHover: colors.accent.darkTeal,
  linkUnderline: colors.primary.teal,
  
  // Backgrounds
  bgPage: colors.secondary.white,             // CHANGED: White, not cyan
  bgHero: colors.primary.gold,                // Gold banner section
  bgContent: colors.secondary.white,
  bgSection: colors.secondary.lightGray,
  
  // Text
  textPrimary: colors.neutral.black,
  textSecondary: colors.neutral.mediumGray,
  textHeading: colors.neutral.black,          // CHANGED: Black headings
  
  // Icons & Badges
  iconPrimary: colors.accent.darkTeal,        // Appliance icons
  iconSecondary: colors.primary.teal,
  checkmark: colors.neutral.black,            // Black checkmarks
  
  // Accents & Badges
  alert: colors.accent.brightRed,
  promotion: colors.accent.orange,            // Orange for deals
  rating: colors.accent.orange,
  premium: colors.accent.magenta,
  badge: colors.primary.teal,
  
  // Status indicators
  success: colors.accent.green,
  warning: colors.accent.orange,
  error: colors.accent.brightRed,
  
  // Borders
  border: colors.neutral.borderGray,
  borderDark: colors.neutral.mediumGray,
  
  // Chat specific (adjusted for PartSelect)
  userMessage: colors.primary.teal,
  assistantMessage: colors.secondary.lightGray,
  chatBackground: colors.secondary.white,     // White, not cyan
  inputBorder: colors.neutral.borderGray,
  inputFocus: colors.primary.teal,
};