"""
Invoice Generator Service for YTEmpire
Handles invoice generation, PDF creation, and billing document management
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

from sqlalchemy import select, and_
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.models.subscription import Subscription
from app.models.billing import Invoice, InvoiceItem, PaymentHistory
from app.models.cost import Cost
from app.core.config import settings

logger = logging.getLogger(__name__)


class InvoiceGenerator:
    """
    Service for generating invoices and billing documents
    """
    
    def __init__(self):
        self.invoice_dir = Path(settings.UPLOAD_DIR) / "invoices"
        self.invoice_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for invoices"""
        self.styles.add(ParagraphStyle(
            name='InvoiceTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='InvoiceHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#333333')
        ))
        
        self.styles.add(ParagraphStyle(
            name='InvoiceRight',
            parent=self.styles['Normal'],
            alignment=TA_RIGHT
        ))
    
    async def generate_invoice(
        self,
        user_id: str,
        subscription_id: str,
        billing_period_start: datetime,
        billing_period_end: datetime,
        invoice_type: str = "subscription"
    ) -> Dict[str, Any]:
        """
        Generate invoice for user
        
        Args:
            user_id: User ID
            subscription_id: Subscription ID
            billing_period_start: Start of billing period
            billing_period_end: End of billing period
            invoice_type: Type of invoice (subscription, usage, credit)
        
        Returns:
            Invoice details with file path
        """
        try:
            async with AsyncSessionLocal() as db:
                # Get user and subscription details
                user = await db.get(User, user_id)
                subscription = await db.get(Subscription, subscription_id)
                
                if not user or not subscription:
                    raise ValueError("User or subscription not found")
                
                # Generate invoice number
                invoice_number = self._generate_invoice_number()
                
                # Calculate charges
                invoice_data = await self._calculate_invoice_amounts(
                    db, user_id, subscription, billing_period_start, billing_period_end
                )
                
                # Create invoice record
                invoice = Invoice(
                    id=str(uuid.uuid4()),
                    invoice_number=invoice_number,
                    user_id=user_id,
                    subscription_id=subscription_id,
                    invoice_type=invoice_type,
                    billing_period_start=billing_period_start,
                    billing_period_end=billing_period_end,
                    subtotal=invoice_data['subtotal'],
                    tax_amount=invoice_data['tax'],
                    total_amount=invoice_data['total'],
                    currency='USD',
                    status='draft',
                    due_date=datetime.utcnow() + timedelta(days=30)
                )
                db.add(invoice)
                
                # Add invoice items
                for item in invoice_data['items']:
                    invoice_item = InvoiceItem(
                        id=str(uuid.uuid4()),
                        invoice_id=invoice.id,
                        description=item['description'],
                        quantity=item['quantity'],
                        unit_price=item['unit_price'],
                        amount=item['amount']
                    )
                    db.add(invoice_item)
                
                await db.commit()
                
                # Generate PDF
                pdf_path = await self._generate_pdf(
                    invoice, user, subscription, invoice_data
                )
                
                # Update invoice with PDF path
                invoice.pdf_url = pdf_path
                invoice.status = 'sent'
                await db.commit()
                
                return {
                    'success': True,
                    'invoice_id': invoice.id,
                    'invoice_number': invoice_number,
                    'pdf_path': pdf_path,
                    'total_amount': float(invoice.total_amount),
                    'due_date': invoice.due_date.isoformat()
                }
        
        except Exception as e:
            logger.error(f"Invoice generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _calculate_invoice_amounts(
        self,
        db,
        user_id: str,
        subscription: Any,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate invoice amounts including usage charges"""
        items = []
        
        # Base subscription charge
        items.append({
            'description': f"{subscription.plan_name} Subscription",
            'quantity': 1,
            'unit_price': float(subscription.monthly_price),
            'amount': float(subscription.monthly_price)
        })
        
        # Calculate usage charges
        usage_charges = await self._calculate_usage_charges(
            db, user_id, start_date, end_date
        )
        
        if usage_charges['video_generation'] > 0:
            items.append({
                'description': f"Video Generation Charges",
                'quantity': usage_charges['video_count'],
                'unit_price': usage_charges['video_generation'] / usage_charges['video_count'],
                'amount': usage_charges['video_generation']
            })
        
        if usage_charges['ai_costs'] > 0:
            items.append({
                'description': "AI Service Usage",
                'quantity': 1,
                'unit_price': usage_charges['ai_costs'],
                'amount': usage_charges['ai_costs']
            })
        
        # Calculate totals
        subtotal = sum(item['amount'] for item in items)
        tax_rate = 0.1  # 10% tax
        tax = subtotal * tax_rate
        total = subtotal + tax
        
        return {
            'items': items,
            'subtotal': Decimal(str(subtotal)),
            'tax': Decimal(str(tax)),
            'total': Decimal(str(total))
        }
    
    async def _calculate_usage_charges(
        self,
        db,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Calculate usage-based charges"""
        # Get video generation costs
        video_costs = await db.execute(
            select(Cost).where(
                and_(
                    Cost.user_id == user_id,
                    Cost.created_at >= start_date,
                    Cost.created_at <= end_date,
                    Cost.operation == 'video_generation'
                )
            )
        )
        
        video_charges = sum(cost.amount for cost in video_costs.scalars())
        video_count = video_costs.scalars().count() if video_costs else 0
        
        # Get AI service costs
        ai_costs = await db.execute(
            select(Cost).where(
                and_(
                    Cost.user_id == user_id,
                    Cost.created_at >= start_date,
                    Cost.created_at <= end_date,
                    Cost.service.in_(['openai', 'elevenlabs', 'dalle'])
                )
            )
        )
        
        ai_charges = sum(cost.amount for cost in ai_costs.scalars())
        
        return {
            'video_generation': float(video_charges),
            'video_count': video_count,
            'ai_costs': float(ai_charges)
        }
    
    async def _generate_pdf(
        self,
        invoice: Any,
        user: Any,
        subscription: Any,
        invoice_data: Dict[str, Any]
    ) -> str:
        """Generate PDF invoice file"""
        filename = f"invoice_{invoice.invoice_number}.pdf"
        filepath = self.invoice_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Header
        story.append(Paragraph("INVOICE", self.styles['InvoiceTitle']))
        story.append(Spacer(1, 0.5 * inch))
        
        # Company info
        company_info = [
            ["YTEmpire", f"Invoice #: {invoice.invoice_number}"],
            ["123 AI Street", f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}"],
            ["Tech City, TC 12345", f"Due Date: {invoice.due_date.strftime('%Y-%m-%d')}"],
            ["support@ytempire.ai", ""]
        ]
        
        company_table = Table(company_info, colWidths=[3.5 * inch, 3.5 * inch])
        company_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        story.append(company_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Bill to
        story.append(Paragraph("<b>Bill To:</b>", self.styles['InvoiceHeader']))
        bill_to = [
            [user.full_name or user.email],
            [user.company_name or ""],
            [user.email]
        ]
        
        for line in bill_to:
            if line[0]:
                story.append(Paragraph(line[0], self.styles['Normal']))
        
        story.append(Spacer(1, 0.3 * inch))
        
        # Invoice items table
        items_data = [['Description', 'Quantity', 'Unit Price', 'Amount']]
        
        for item in invoice_data['items']:
            items_data.append([
                item['description'],
                str(item['quantity']),
                f"${item['unit_price']:.2f}",
                f"${item['amount']:.2f}"
            ])
        
        # Add totals
        items_data.append(['', '', 'Subtotal:', f"${float(invoice_data['subtotal']):.2f}"])
        items_data.append(['', '', 'Tax (10%):', f"${float(invoice_data['tax']):.2f}"])
        items_data.append(['', '', 'Total:', f"${float(invoice_data['total']):.2f}"])
        
        items_table = Table(items_data, colWidths=[3.5 * inch, 1 * inch, 1.25 * inch, 1.25 * inch])
        items_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -4), colors.beige),
            ('GRID', (0, 0), (-1, -4), 1, colors.black),
            ('FONTNAME', (2, -3), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (2, -3), (-1, -1), 11),
        ]))
        story.append(items_table)
        story.append(Spacer(1, 0.5 * inch))
        
        # Payment terms
        story.append(Paragraph("<b>Payment Terms:</b>", self.styles['InvoiceHeader']))
        story.append(Paragraph("Payment is due within 30 days of invoice date.", self.styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Thank you for your business!", self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return f"/invoices/{filename}"
    
    def _generate_invoice_number(self) -> str:
        """Generate unique invoice number"""
        timestamp = datetime.utcnow().strftime("%Y%m")
        random_suffix = str(uuid.uuid4())[:6].upper()
        return f"INV-{timestamp}-{random_suffix}"
    
    async def get_invoice(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Get invoice details"""
        async with AsyncSessionLocal() as db:
            invoice = await db.get(Invoice, invoice_id)
            if not invoice:
                return None
            
            # Get invoice items
            items_result = await db.execute(
                select(InvoiceItem).where(InvoiceItem.invoice_id == invoice_id)
            )
            items = items_result.scalars().all()
            
            return {
                'invoice_id': invoice.id,
                'invoice_number': invoice.invoice_number,
                'user_id': invoice.user_id,
                'status': invoice.status,
                'total_amount': float(invoice.total_amount),
                'due_date': invoice.due_date.isoformat(),
                'pdf_url': invoice.pdf_url,
                'items': [
                    {
                        'description': item.description,
                        'quantity': item.quantity,
                        'unit_price': float(item.unit_price),
                        'amount': float(item.amount)
                    }
                    for item in items
                ]
            }
    
    async def list_user_invoices(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List user's invoices"""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Invoice)
                .where(Invoice.user_id == user_id)
                .order_by(Invoice.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            
            invoices = result.scalars().all()
            
            return [
                {
                    'invoice_id': inv.id,
                    'invoice_number': inv.invoice_number,
                    'total_amount': float(inv.total_amount),
                    'status': inv.status,
                    'due_date': inv.due_date.isoformat(),
                    'created_at': inv.created_at.isoformat(),
                    'pdf_url': inv.pdf_url
                }
                for inv in invoices
            ]
    
    async def mark_invoice_paid(
        self,
        invoice_id: str,
        payment_method: str,
        transaction_id: str
    ) -> bool:
        """Mark invoice as paid"""
        async with AsyncSessionLocal() as db:
            invoice = await db.get(Invoice, invoice_id)
            if not invoice:
                return False
            
            invoice.status = 'paid'
            invoice.paid_at = datetime.utcnow()
            
            # Create payment record
            payment = PaymentHistory(
                id=str(uuid.uuid4()),
                user_id=invoice.user_id,
                invoice_id=invoice_id,
                amount=invoice.total_amount,
                payment_method=payment_method,
                transaction_id=transaction_id,
                status='completed'
            )
            db.add(payment)
            
            await db.commit()
            return True


# Singleton instance
invoice_generator = InvoiceGenerator()