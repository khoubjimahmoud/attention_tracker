Attention Tracker

A web application to track user attention percentages using Django, integrating real-time frame analysis and user detail visualization.
Table of Contents

    Project Overview
    Features
    Technologies Used
    Installation
    Usage
    Project Structure
    Future Enhancements
    Contributing
    Contact

Project Overview

The Attention Tracker is designed to analyze and track user attention percentages based on real-time frame analysis. The results are displayed in a clean, user-friendly interface with attention percentage calculations shown to two decimal places.
Features

    User Attention Calculation: Tracks and displays attention percentage for each user.
    Dynamic Visualization: View detailed attention analysis for each user.
    Clean UI: Utilizes Bulma CSS framework for a responsive and modern interface.

Technologies Used

    Backend: Django 5.0.7
    Frontend: HTML5, Bulma (CSS3), JavaScript
    Data Analysis: Python for percentage calculations and frame extraction
    Database: SQLite (default for Django)
    Machine Learning Model: CNN 

Installation

Follow these steps to get the project running locally:

    Clone the repository:

bash

    git clone <your-repo-link>
    cd attention_tracker

Create and activate a virtual environment:

bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required dependencies:

bash

    pip install -r requirements.txt

Set up the database and static files:

bash

    python manage.py migrate
    python manage.py collectstatic

Run the development server:

bash

    python manage.py runserver

    Access the application: Open your browser and navigate to http://127.0.0.1:8000/.

Usage

    The home page displays a table with user names and their corresponding attention percentages.
    Click the "View Details" button to see more detailed data for each user.
    The percentages are automatically calculated and updated in real-time.

Project Structure

Here's an overview of the key directories and files in this project:

php

attention_tracker/
│
├── tracker/                # Django app containing views, models, templates, and static files
|   ├── migrations/
│   ├── templates/
│   │   ├── home.html       # Main template for displaying attention data
│   └── static/
│       ├── css/            # Custom CSS (if any)
│       └── js/             # Custom JavaScript (if any)
├── manage.py               # Django management script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

Future Enhancements

    Live Tracking: Implement real-time frame extraction and analysis for more interactive results.
    Advanced Analytics: Include charts and graphs to better visualize attention trends.
    User Authentication: Add login and user-specific data tracking.
    Database Switch: Move from SQLite to PostgreSQL for better scalability.

Contributing

Contributions are welcome! Please follow these steps:

    Fork the project.
    Create your feature branch: git checkout -b feature/YourFeature.
    Commit your changes: git commit -m 'Add YourFeature'.
    Push to the branch: git push origin feature/YourFeature.
    Open a pull request.


Contact

For questions or feedback, feel free to reach out:

    Mahmoud KHOUBJI
    mahmoudkhoubji25@gmail.com
    LinkedIn : Mahmoud Khoubji