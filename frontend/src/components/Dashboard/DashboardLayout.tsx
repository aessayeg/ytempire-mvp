import React, { useState } from 'react';
import {  Link, Outlet, useLocation  } from 'react-router-dom';
import {  clsx  } from 'clsx';
import {  useAuthStore  } from '../../stores/authStore';

const navigation = [ { name: 'Overview', href: '/dashboard', icon: '📊' },
  { name: 'Channels', href: '/dashboard/channels', icon: '📺' },
  { name: 'Videos', href: '/dashboard/videos', icon: '🎬' },
  { name: 'Analytics', href: '/dashboard/analytics', icon: '📈' },
  { name: 'Revenue', href: '/dashboard/revenue', icon: '💰' },
  { name: 'Settings', href: '/dashboard/settings', icon: '⚙️' } ];

export const DashboardLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const { user, logout } = useAuthStore();

  return (
    <>
      <div className="min-h-screen bg-gray-50">
      {/* Mobile sidebar toggle */}
      <div className="lg:hidden">
        <button
          type="button"
          className="fixed top-4 left-4 z-50 p-2 rounded-md bg-white shadow-lg"
          onClick={() => setSidebarOpen(!sidebarOpen}
        >
          <span className="sr-only">Open sidebar</span>
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6 h16 M4 12 h16 M4 18 h16" />
          </svg>
        </button>
      </div>

      {/* Sidebar */}
      <div
        className={ clsx(
          'fixed inset-y-0 left-0 z-40 w-64 bg-white border-r border-gray-200 transform transition-transform duration-200 ease-in-out, lg:translate-x-0',
          {
            'translate-x-0': sidebarOpen,
            '-translate-x-full': !sidebarOpen }
        )}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-center h-16px-4 border-b border-gray-200">
            <h1 className="text-2 xl font-bold text-primary-600">YTEmpire</h1>
          </div>

          {/* Navigation */}
          <nav className="flex-1px-4 py-4 space-y-1">
            {navigation.map((_(item) => {
              const isActive = location.pathname === item.href;
              return (
    <>
      <Link
                  key={item.name}
                  to={item.href}
                  className={ clsx(
                    'flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                    {
                      'bg-primary-100 text-primary-900': isActive,
                      'text-gray-600, hover:bg-gray-50, hover:text-gray-900': !isActive }
                  )}
                >
                  <span className="mr-3 text-lg">{item.icon}</span>
                  {item.name}
                </Link>
              )});
}
          </nav>

          {/* User info */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-full bg-primary-500 flex items-center justify-center text-white font-semibold">
                  {user?.username?.[0]?.toUpperCase() || 'U'}
                </div>
              </div>
      <div className="ml-3 flex-1">
                <p className="text-sm font-medium text-gray-900">{user?.username}</p>
                <p className="text-xs text-gray-500">{user?.subscription_tier} Plan</p>
              </div>
              <button
                onClick={logout}
                className="ml-2 p-1 rounded-md text-gray-400, hover:text-gray-600"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16 l4-4 m0 0 l-4-4 m4 4 H7 m6 4 v1 a3 3 0 01-3 3 H6 a3 3 0 01-3-3 V7 a3 3 0 013-3 h4 a3 3 0 013 3 v1" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <header className="bg-white border-b border-gray-200">
          <div className="px-4, sm:px-6, lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <h2 className="text-2 xl font-semibold text-gray-900">
                {navigation.find(item => item.href === location.pathname)?.name || 'Dashboard'}
              </h2>
              
              {/* Stats bar */}
              <div className="flex items-center space-x-6">
                <div className="text-sm">
                  <span className="text-gray-500">Videos Today:</span>
                  <span className="ml-2 font-semibold text-gray-900">
                    {user?.total_videos_generated || 0} / {user?.videos_per_day_limit || 5}
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-500">Channels:</span>
                  <span className="ml-2 font-semibold text-gray-900">
                    0 / {user?.channels_limit || 1}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 sm:p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  </>
  )};