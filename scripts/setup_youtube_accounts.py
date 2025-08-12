"""
Setup Script for YouTube Multi-Account Configuration
Helps configure all 15 YouTube accounts for quota distribution
"""
import os
import sys
import json
import time
from typing import List, Dict
import webbrowser

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.youtube_oauth_service import youtube_oauth_service

def setup_youtube_accounts():
    """Interactive setup for YouTube accounts"""
    print("\n" + "="*60)
    print("  YTEmpire YouTube Multi-Account Setup")
    print("="*60)
    
    # Check for Google OAuth credentials
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("\n⚠️  Google OAuth credentials not found in environment!")
        print("\nPlease set the following environment variables:")
        print("  - GOOGLE_CLIENT_ID")
        print("  - GOOGLE_CLIENT_SECRET")
        print("\nYou can get these from Google Cloud Console:")
        print("1. Go to https://console.cloud.google.com")
        print("2. Create a new project or select existing")
        print("3. Enable YouTube Data API v3")
        print("4. Create OAuth 2.0 credentials")
        print("5. Add http://localhost:8000/api/v1/youtube/oauth/callback to redirect URIs")
        return
    
    # Load current account status
    youtube_oauth_service.load_accounts()
    accounts = youtube_oauth_service.accounts
    
    # Show current status
    print(f"\nTotal accounts configured: {len(accounts)}")
    
    authorized = []
    unauthorized = []
    
    for account_id, account in accounts.items():
        if account.get('refresh_token'):
            authorized.append(account)
        else:
            unauthorized.append(account)
    
    print(f"✅ Authorized: {len(authorized)}")
    print(f"❌ Unauthorized: {len(unauthorized)}")
    
    if authorized:
        print("\nAuthorized accounts:")
        for acc in authorized:
            print(f"  - {acc['account_id']}: {acc.get('channel_name', 'Unknown')}")
    
    if not unauthorized:
        print("\n✨ All accounts are authorized!")
        return
    
    print(f"\n{len(unauthorized)} accounts need authorization:")
    for acc in unauthorized:
        print(f"  - {acc['account_id']}: {acc.get('email', 'Not set')}")
    
    # Ask if user wants to authorize accounts
    response = input("\nDo you want to authorize these accounts now? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    # Generate auth URLs
    print("\n🔐 Generating authorization URLs...")
    auth_urls = []
    
    for account in unauthorized:
        try:
            auth_url = youtube_oauth_service.get_auth_url(account['account_id'])
            auth_urls.append({
                'account_id': account['account_id'],
                'email': account.get('email'),
                'url': auth_url
            })
            print(f"  ✓ {account['account_id']}")
        except Exception as e:
            print(f"  ✗ {account['account_id']}: {e}")
    
    if not auth_urls:
        print("\n❌ Failed to generate authorization URLs")
        return
    
    # Save URLs to file
    urls_file = "youtube_auth_urls.txt"
    with open(urls_file, 'w') as f:
        f.write("YouTube Account Authorization URLs\n")
        f.write("="*50 + "\n\n")
        for item in auth_urls:
            f.write(f"Account: {item['account_id']}\n")
            f.write(f"Email: {item['email']}\n")
            f.write(f"URL: {item['url']}\n")
            f.write("-"*50 + "\n\n")
    
    print(f"\n📄 Authorization URLs saved to: {urls_file}")
    
    # Option to open URLs in browser
    print("\nOptions:")
    print("1. Open all URLs in browser (automated)")
    print("2. Open URLs one by one (manual)")
    print("3. Copy URLs from file (manual)")
    
    choice = input("\nSelect option (1/2/3): ")
    
    if choice == '1':
        print("\n🌐 Opening all URLs in browser...")
        print("Please authorize each account and complete the OAuth flow.")
        for item in auth_urls:
            webbrowser.open(item['url'])
            time.sleep(2)  # Delay between opens
    
    elif choice == '2':
        print("\n🌐 Opening URLs one by one...")
        for item in auth_urls:
            print(f"\nAccount: {item['account_id']}")
            print(f"Email: {item['email']}")
            input("Press Enter to open authorization URL...")
            webbrowser.open(item['url'])
            input("Press Enter after completing authorization...")
    
    else:
        print(f"\n📋 Please open {urls_file} and manually visit each URL")
        print("Complete the OAuth flow for each account.")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Complete OAuth authorization for each account")
    print("2. Verify accounts at: http://localhost:8000/api/v1/youtube/accounts/authorized")
    print("3. Test video upload with: python scripts/test_youtube_upload.py")

def check_account_status():
    """Check current status of all YouTube accounts"""
    youtube_oauth_service.load_accounts()
    
    print("\n" + "="*60)
    print("  YouTube Account Status")
    print("="*60)
    
    for account_id, account in youtube_oauth_service.accounts.items():
        status = "✅ Authorized" if account.get('refresh_token') else "❌ Not Authorized"
        channel = account.get('channel_name', 'Not Connected')
        print(f"\n{account_id}:")
        print(f"  Status: {status}")
        print(f"  Email: {account.get('email', 'Not Set')}")
        print(f"  Channel: {channel}")
        
        if account.get('refresh_token'):
            # Check health
            try:
                health = youtube_oauth_service.check_account_health(account_id)
                print(f"  Health: {health.get('status', 'Unknown')}")
            except Exception as e:
                print(f"  Health: Error - {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Multi-Account Setup")
    parser.add_argument('--status', action='store_true', help='Check account status')
    parser.add_argument('--setup', action='store_true', help='Run setup wizard')
    
    args = parser.parse_args()
    
    if args.status:
        check_account_status()
    else:
        setup_youtube_accounts()