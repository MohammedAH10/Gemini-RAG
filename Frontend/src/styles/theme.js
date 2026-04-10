export const theme = {
  colors: {
    navyDeep: '#0a1628',
    navyMid: '#0f2140',
    navyLight: '#1a2d50',
    blueElectric: '#4facfe',
    blueGlow: '#00f2fe',
    purpleDeep: '#7c3aed',
    purpleLight: '#a78bfa',
    pinkLight: '#f9a8d4',
    pinkGlow: '#f472b6',
    textPrimary: '#e8f0fe',
    textSecondary: '#94a3b8',
    textMuted: '#64748b',
  },
  gradients: {
    primary: 'linear-gradient(135deg, #4facfe 0%, #7c3aed 100%)',
    glow: 'linear-gradient(135deg, #00f2fe 0%, #f9a8d4 100%)',
    card: 'linear-gradient(180deg, rgba(26, 45, 80, 0.6) 0%, rgba(15, 33, 64, 0.8) 100%)',
  },
  shadows: {
    glow: '0 0 20px rgba(79, 172, 254, 0.15)',
    card: '0 4px 24px rgba(0, 0, 0, 0.3)',
    glowPurple: '0 0 20px rgba(124, 58, 237, 0.2)',
  },
  radius: {
    sm: '0.375rem',
    md: '0.75rem',
    lg: '1rem',
    xl: '1.5rem',
    full: '9999px',
  },
  transitions: {
    fast: '150ms ease',
    base: '250ms ease',
    slow: '400ms ease',
  },
};

export const glassCard = {
  background: 'rgba(15, 33, 64, 0.6)',
  backdropFilter: 'blur(12px)',
  border: '1px solid rgba(79, 172, 254, 0.15)',
  borderRadius: 'var(--radius-lg)',
};

export const glowButton = {
  background: 'linear-gradient(135deg, #4facfe 0%, #7c3aed 100%)',
  boxShadow: '0 0 20px rgba(79, 172, 254, 0.3)',
};
