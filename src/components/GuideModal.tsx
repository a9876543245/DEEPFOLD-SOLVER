import React from 'react';
import {
  X, Users, Spade, Settings, BarChart3, GitBranch,
  Target, Wrench, MousePointer, ChevronRight, Lightbulb
} from 'lucide-react';
import { useT } from '../lib/i18n';

interface Props {
  onClose: () => void;
}

interface StepProps {
  num: number;
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}

function Step({ num, icon, title, children }: StepProps) {
  return (
    <div style={{
      display: 'flex', gap: 16, padding: 18,
      background: 'var(--color-glass)', borderRadius: 12,
      border: '1px solid var(--color-glass-border)',
    }}>
      {/* Step number badge */}
      <div style={{
        width: 40, height: 40, borderRadius: '50%', flexShrink: 0,
        background: 'linear-gradient(135deg, #0A84FF, #BF5AF2)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 18, fontWeight: 800, color: '#fff',
      }}>{num}</div>

      <div style={{ flex: 1 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8,
        }}>
          <span style={{ color: 'var(--color-accent)' }}>{icon}</span>
          <span style={{ fontSize: 16, fontWeight: 700 }}>{title}</span>
        </div>
        <div style={{ fontSize: 13, color: 'var(--color-text-secondary)', lineHeight: 1.7 }}>
          {children}
        </div>
      </div>
    </div>
  );
}

function Kbd({ children }: { children: string }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      padding: '1px 6px', borderRadius: 4, fontSize: 11, fontWeight: 700,
      background: 'var(--color-bg-tertiary)', border: '1px solid var(--color-glass-border)',
      fontFamily: 'var(--font-mono, monospace)', color: 'var(--color-text-primary)',
    }}>{children}</span>
  );
}

function Tag({ color, children }: { color: string; children: string }) {
  return (
    <span style={{
      fontSize: 12, fontWeight: 700, padding: '1px 6px', borderRadius: 4,
      background: color + '22', color: color,
    }}>{children}</span>
  );
}

export function GuideModal({ onClose }: Props) {
  const t = useT();
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.7)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 20,
    }}>
      <div className="animate-slide-up" style={{
        width: '100%', maxWidth: 720, maxHeight: '90vh',
        background: 'var(--color-bg-secondary)',
        border: '1px solid var(--color-border)',
        borderRadius: 16, display: 'flex', flexDirection: 'column',
        boxShadow: '0 32px 80px rgba(0,0,0,0.5)',
        overflow: 'hidden',
      }}>

        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '18px 24px', borderBottom: '1px solid var(--color-border)',
          background: 'var(--color-bg-primary)', flexShrink: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: 'linear-gradient(135deg, #0A84FF, #BF5AF2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 16, fontWeight: 800, color: '#fff',
            }}>?</div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{t('guide.title')}</div>
              <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                {t('guide.subtitle')}
              </div>
            </div>
          </div>
          <button onClick={onClose} style={{
            padding: 8, border: 'none', background: 'var(--color-glass)',
            borderRadius: 8, cursor: 'pointer', color: 'var(--color-text-secondary)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <X size={18} />
          </button>
        </div>

        {/* Scrollable Content */}
        <div style={{
          flex: 1, overflowY: 'auto', padding: '20px 24px',
          display: 'flex', flexDirection: 'column', gap: 14,
        }}>

          {/* Quick intro */}
          <div style={{
            padding: '14px 18px', borderRadius: 10,
            background: 'linear-gradient(135deg, rgba(10,132,255,0.1), rgba(191,90,242,0.1))',
            border: '1px solid rgba(10,132,255,0.2)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <Lightbulb size={16} style={{ color: '#FFD60A' }} />
              <span style={{ fontSize: 14, fontWeight: 700 }}>{t('guide.quickStart')}</span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
              {t('guide.quickStartDesc')}
            </div>
          </div>

          {/* Steps */}
          <Step num={1} icon={<Users size={18} />} title={t('guide.step1.title')}>
            <p>{t('guide.step1.desc')}</p>
          </Step>

          <Step num={2} icon={<Spade size={18} />} title={t('guide.step2.title')}>
            <p>{t('guide.step2.desc')}</p>
          </Step>

          <Step num={3} icon={<Settings size={18} />} title={t('guide.step3.title')}>
            <p>{t('guide.step3.desc')}</p>
          </Step>

          <Step num={4} icon={<BarChart3 size={18} />} title={t('guide.step4.title')}>
            <p>{t('guide.step4.desc')}</p>
          </Step>

          <Step num={5} icon={<GitBranch size={18} />} title={t('guide.step5.title')}>
            <p>{t('guide.step5.desc')}</p>
          </Step>

          <Step num={6} icon={<Target size={18} />} title={t('guide.step6.title')}>
            <p>{t('guide.step6.desc')}</p>
          </Step>

          <Step num={7} icon={<Wrench size={18} />} title={t('guide.step7.title')}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4 }}>
              <div>
                <div style={{ fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: 2, fontSize: 13 }}>
                  <MousePointer size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
                  Range Editor
                </div>
                <span>{t('guide.step7.range')}</span>
              </div>
              <div>
                <div style={{ fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: 2, fontSize: 13 }}>
                  <MousePointer size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
                  Node Locking
                </div>
                <span>{t('guide.step7.lock')}</span>
              </div>
              <div>
                <div style={{ fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: 2, fontSize: 13 }}>
                  <MousePointer size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
                  Heatmap Mode
                </div>
                <span>{t('guide.step7.heatmap')}</span>
              </div>
            </div>
          </Step>

        </div>

        {/* Footer */}
        <div style={{
          padding: '14px 24px', borderTop: '1px solid var(--color-border)',
          background: 'var(--color-bg-primary)', flexShrink: 0,
          display: 'flex', justifyContent: 'center',
        }}>
          <button onClick={onClose} style={{
            padding: '10px 40px', borderRadius: 10, border: 'none',
            background: 'var(--color-accent)', color: '#fff',
            fontSize: 15, fontWeight: 700, cursor: 'pointer',
            fontFamily: 'inherit',
            transition: 'all 200ms ease',
          }}
            onMouseOver={e => { e.currentTarget.style.background = 'var(--color-accent-hover)'; e.currentTarget.style.boxShadow = '0 4px 16px rgba(10,132,255,0.4)'; }}
            onMouseOut={e => { e.currentTarget.style.background = 'var(--color-accent)'; e.currentTarget.style.boxShadow = 'none'; }}
          >
            {t('guide.gotIt')}
          </button>
        </div>
      </div>
    </div>
  );
}
