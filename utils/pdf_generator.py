from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import numpy as np

def generate_enhanced_pdf(model_name, description, results, model_id, doc_path):
    """
    Generate an enhanced PDF documentation for the ML model training results.
    
    Args:
        model_name: Name of the model
        description: Model description
        results: Complete results from ML pipeline
        model_id: Unique model identifier
        doc_path: Path to save the PDF
    """
    doc = SimpleDocTemplate(doc_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue,
        borderWidth=1,
        borderColor=colors.darkblue,
        borderPadding=5
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.darkgreen
    )
    
    # Title Page
    story.append(Paragraph("Machine Learning Model Report", title_style))
    story.append(Spacer(1, 20))
    
    # Model Information Table
    model_info = [
        ['Model Name:', model_name],
        ['Description:', description],
        ['Model ID:', model_id],
        ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Best Model:', results['model_results']['best_model']],
        ['Best Accuracy:', f"{results['model_results']['best_accuracy']:.4f}"]
    ]
    
    model_table = Table(model_info, colWidths=[2*inch, 4*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(model_table)
    story.append(Spacer(1, 30))
    
    # Dataset Summary
    story.append(Paragraph("Dataset Summary", heading_style))
    story.append(Spacer(1, 12))
    
    data_summary = results['data_summary']
    dataset_info = [
        ['Total Rows:', str(data_summary['rows'])],
        ['Total Columns:', str(data_summary['columns'])],
        ['Feature Columns:', str(len(data_summary['features']))],
        ['Target Column:', str(data_summary['target_column'])],
        ['Missing Values:', 'Yes' if any(data_summary['missing_values'].values()) else 'No']
    ]
    
    if data_summary.get('target_classes'):
        dataset_info.append(['Target Classes:', str(len(data_summary['target_classes']))])
    
    dataset_table = Table(dataset_info, colWidths=[2*inch, 4*inch])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(dataset_table)
    story.append(Spacer(1, 20))
    
    # Features List
    story.append(Paragraph("Feature Columns", subheading_style))
    features_text = ", ".join(data_summary['features'])
    story.append(Paragraph(features_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Missing Values Details
    if any(data_summary['missing_values'].values()):
        story.append(Paragraph("Missing Values by Column", subheading_style))
        missing_data = [[col, str(count)] for col, count in data_summary['missing_values'].items() if count > 0]
        
        if missing_data:
            missing_table = Table([['Column', 'Missing Count']] + missing_data, 
                                colWidths=[3*inch, 1*inch])
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(missing_table)
    
    story.append(PageBreak())
    
    # Model Training Results
    story.append(Paragraph("Model Training Results", heading_style))
    story.append(Spacer(1, 12))
    
    # Model Performance Comparison
    model_results = results['model_results']
    performance_data = [['Model', 'Accuracy', 'CV Mean', 'CV Std']]
    
    for model_name_key, model_result in model_results.items():
        if model_name_key not in ['best_model', 'best_accuracy']:
            performance_data.append([
                model_name_key.replace('_', ' ').title(),
                f"{model_result['accuracy']:.4f}",
                f"{model_result['cv_mean']:.4f}",
                f"{model_result['cv_std']:.4f}"
            ])
    
    performance_table = Table(performance_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    performance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(performance_table)
    story.append(Spacer(1, 20))
    
    # Best Model Details
    best_model_name = model_results['best_model']
    best_model_result = model_results[best_model_name]
    
    story.append(Paragraph(f"Best Model: {best_model_name.replace('_', ' ').title()}", subheading_style))
    
    best_model_info = [
        ['Accuracy:', f"{best_model_result['accuracy']:.4f}"],
        ['Cross-Validation Mean:', f"{best_model_result['cv_mean']:.4f}"],
        ['Cross-Validation Std:', f"{best_model_result['cv_std']:.4f}"],
        ['Model Type:', best_model_name.replace('_', ' ').title()]
    ]
    
    best_model_table = Table(best_model_info, colWidths=[2*inch, 2*inch])
    best_model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(best_model_table)
    story.append(Spacer(1, 20))
    
    # Classification Report for Best Model
    if 'classification_report' in best_model_result:
        story.append(Paragraph("Detailed Classification Report", subheading_style))
        
        class_report = best_model_result['classification_report']
        
        # Create classification report table
        report_data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
        
        for class_name, metrics in class_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                if isinstance(metrics, dict):
                    report_data.append([
                        str(class_name),
                        f"{metrics['precision']:.3f}",
                        f"{metrics['recall']:.3f}",
                        f"{metrics['f1-score']:.3f}",
                        str(int(metrics['support']))
                    ])
        
        # Add summary rows
        if 'macro avg' in class_report:
            macro_avg = class_report['macro avg']
            report_data.append([
                'Macro Avg',
                f"{macro_avg['precision']:.3f}",
                f"{macro_avg['recall']:.3f}",
                f"{macro_avg['f1-score']:.3f}",
                str(int(macro_avg['support']))
            ])
        
        if 'weighted avg' in class_report:
            weighted_avg = class_report['weighted avg']
            report_data.append([
                'Weighted Avg',
                f"{weighted_avg['precision']:.3f}",
                f"{weighted_avg['recall']:.3f}",
                f"{weighted_avg['f1-score']:.3f}",
                str(int(weighted_avg['support']))
            ])
        
        report_table = Table(report_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        report_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.lightgrey]),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue)
        ]))
        
        story.append(report_table)
        story.append(Spacer(1, 20))
    
    # Feature Importance
    if 'feature_importance' in best_model_result:
        story.append(Paragraph("Feature Importance", heading_style))
        fi_data = best_model_result['feature_importance']
        
        # Create feature importance table
        fi_table_data = [['Feature', 'Importance']]
        for feature, importance in fi_data.items():
            fi_table_data.append([feature, f"{importance:.4f}"])
        
        fi_table = Table(fi_table_data, colWidths=[3*inch, 2*inch])
        fi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(fi_table)
        story.append(Spacer(1, 20))
    
    # Hyperparameters Tuning Results
    if 'hyperparameters' in best_model_result:
        story.append(Paragraph("Hyperparameters Tuning Results", heading_style))
        tuning_results = best_model_result['hyperparameters']
        
        # Create hyperparameters table
        tuning_table_data = [['Parameter', 'Value']]
        for param, value in tuning_results.items():
            tuning_table_data.append([param, str(value)])
        
        tuning_table = Table(tuning_table_data, colWidths=[3*inch, 2*inch])
        tuning_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(tuning_table)
        story.append(Spacer(1, 20))
    
    # Confusion Matrix
    if 'confusion_matrix' in best_model_result:
        story.append(Paragraph("Confusion Matrix", heading_style))
        cm_data = best_model_result['confusion_matrix']
        
        # Create confusion matrix table
        cm_table_data = [[''] + [str(i) for i in range(len(cm_data))]]
        for i, row in enumerate(cm_data):
            cm_table_data.append([str(i)] + [str(int(val)) for val in row])
        
        cm_table = Table(cm_table_data, colWidths=[0.5*inch] + [0.3*inch]*len(cm_data))
        cm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(cm_table)
        story.append(Spacer(1, 20))
    
    # ROC Curve
    if 'roc_curve' in best_model_result:
        story.append(Paragraph("ROC Curve", heading_style))
        roc_curve = best_model_result['roc_curve']
        
        # Create ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(roc_curve['fpr'], roc_curve['tpr'], color='blue')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.grid(True)
        
        # Save plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # Encode plot to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_html = f'<img src="data:image/png;base64,{img_base64}" width="600"/>'
        
        story.append(Paragraph(img_html, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Precision-Recall Curve
    if 'precision_recall_curve' in best_model_result:
        story.append(Paragraph("Precision-Recall Curve", heading_style))
        pr_curve = best_model_result['precision_recall_curve']
        
        # Create Precision-Recall curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(pr_curve['recall'], pr_curve['precision'], color='green')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Save plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # Encode plot to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_html = f'<img src="data:image/png;base64,{img_base64}" width="600"/>'
        
        story.append(Paragraph(img_html, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Residuals Analysis
    if 'residuals' in best_model_result:
        story.append(Paragraph("Residuals Analysis", heading_style))
        residuals = best_model_result['residuals']
        
        # Create residuals plot
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(residuals)), residuals, color='purple')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True)
        
        # Save plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # Encode plot to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_html = f'<img src="data:image/png;base64,{img_base64}" width="600"/>'
        
        story.append(Paragraph(img_html, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Save the PDF document
    doc.build(story)
