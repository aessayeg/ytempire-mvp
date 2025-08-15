import React from 'react';
import { 
  Box,
  Link
 } from '@mui/material';

interface SkipNavigationProps {
  links?: Array<{
    href: string,
  label: string}>;
}

export const SkipNavigation: React.FC<SkipNavigationProps> = ({
  links = [
    { href: '#main-content', label: 'Skip to main content' },
    { href: '#navigation', label: 'Skip to navigation' },
    { href: '#footer', label: 'Skip to footer' }
  ]
}) => {
  return (
    <Box
      component="nav"
      aria-label="Skip navigation"
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        zIndex: 9999,
        '& a': {
          position: 'absolute',
          left: '-9999px',
          top: 'auto',
          width: '1px',
          height: '1px',
          overflow: 'hidden',
          '&:focus': {
            position: 'fixed',
            top: 0,
            left: 0,
            width: 'auto',
            height: 'auto',
            padding: 2,
            backgroundColor: 'primary.main',
            color: 'primary.contrastText',
            textDecoration: 'none',
            zIndex: 10000,
            borderRadius: '0 0 4px 0',

          }
        }
      }}
    >
      {links.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          tabIndex={0}
          onClick={(e) => {
            e.preventDefault();
            const target = document.querySelector(link.href);
            if (target) {
              (target as HTMLElement).focus();
              target.scrollIntoView({ behavior: 'smooth' })}
          }}
        >
          {link.label}
        </Link>
      ))}
    </Box>
  )};