paragraph_analyzer/
│
├── app.py                 # Main Flask application
├── document_parser.py     # Document parsing & paragraph extraction
├── similarity_analyzer.py # Similarity analysis logic
├── database_manager.py    # Database operations
├── requirements.txt       # Project dependencies
├── static/                # Static assets
│   └── css/
│       └── custom.css     # Additional custom styles
│
├── templates/             # HTML templates
│   ├── base.html          # Base template with layout
│   ├── index.html         # Dashboard template
│   ├── paragraphs.html    # Paragraphs viewing template
│   ├── similarity.html    # Similarity analysis template
│   ├── tags.html          # Tag management template
│   ├── 404.html           # Not found error page
│   └── 500.html           # Server error page
│
├── uploads/               # Folder for uploaded documents
├── paragraph_analyzer.db  # SQLite database
└── paragraph_analyzer.log # Application log file
