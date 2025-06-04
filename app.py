from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pymongo
from datetime import datetime, timedelta
import boto3
import json
import os
import logging
from functools import wraps
import nltk
from textblob import TextBlob
import numpy as np
from collections import defaultdict
import uuid
import openai
import anthropic
import google.generativeai as genai
import google
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased"  # Light and fast
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

import re
import random

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# AI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY')
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')

# Initialize AI clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

if GOOGLE_AI_API_KEY:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')

# MongoDB Atlas Configuration
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/username:password@cluster.mongodb.net/mindful_muse?retryWrites=true&w=majority')
# AWS Configuration
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
S3_BUCKET = os.environ.get('S3_BUCKET_NAME', 'mindful-muse-assets')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')

# Initialize AWS clients
s3_client = boto3.client('s3', 
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name=AWS_REGION)

sns_client = boto3.client('sns',
                         aws_access_key_id=AWS_ACCESS_KEY,
                         aws_secret_access_key=AWS_SECRET_KEY,
                         region_name=AWS_REGION)

lambda_client = boto3.client('lambda',
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=AWS_REGION)

# Initialize MongoDB
try:
    mongo_client = pymongo.MongoClient(MONGODB_URI)
    db = mongo_client.mindful_muse
    users_collection = db.users
    reflections_collection = db.reflections
    admin_collection = db.admin_data
    ai_insights_collection = db.ai_insights
    
    # Test connection
    mongo_client.admin.command('ping')
    print("MongoDB connection successful")
except Exception as e:
    print(f"MongoDB connection failed: {e}")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize local AI models for offline capability
try:
    # Load a lightweight emotion detection model
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        tokenizer="j-hartmann/emotion-english-distilroberta-base"
    )
    
    # Load text generation model for prompts
    text_generator = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium",
        tokenizer="microsoft/DialoGPT-medium"
    )
    print("Local AI models loaded successfully")
except Exception as e:
    print(f"Local AI models loading failed: {e}")
    emotion_classifier = None
    text_generator = None

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI Helper Functions
class AIInsightGenerator:
    def __init__(self):
        self.providers = ['openai', 'anthropic', 'gemini', 'local']

    def generate_with_openai(self, prompt, max_tokens=150):
        """Generate content using OpenAI GPT"""
        try:
            if not OPENAI_API_KEY:
                return None
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a compassionate mental health assistant focused on emotional wellness and mindfulness."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return None
    
    def generate_with_anthropic(self, prompt, max_tokens=150):
        """Generate content using Anthropic Claude"""
        try:
            if not ANTHROPIC_API_KEY:
                return None
            
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": f"As a compassionate mental health assistant: {prompt}"}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return None
    
    def generate_with_gemini(self, prompt, max_tokens=150):
        """Generate content using Google Gemini"""
        try:
            if not GOOGLE_AI_API_KEY:
                return None
            
            full_prompt = f"As a compassionate mental health assistant focused on emotional wellness: {prompt}"
            response = gemini_model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return None
    
    def generate_with_local(self, prompt, max_tokens=100):
        """Generate content using local models"""
        try:
            if not text_generator:
                return None
            
            response = text_generator(
                prompt,
                max_length=max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            return response[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            logger.error(f"Local generation error: {e}")
            return None
    
    def generate_insight(self, prompt, preferred_provider='openai', max_tokens=150):
        """Generate AI insight with fallback to other providers"""
        providers = [preferred_provider] + [p for p in self.providers if p != preferred_provider]
        for provider in providers:
            if provider == 'openai':
                result = self.generate_with_openai(prompt, max_tokens=max_tokens)
            elif provider == 'anthropic':
                result = self.generate_with_anthropic(prompt, max_tokens=max_tokens)
            elif provider == 'gemini':
                result = self.generate_with_gemini(prompt, max_tokens=max_tokens)
            elif provider == 'local':
                result = self.generate_with_local(prompt, max_tokens=max_tokens)
            else:
                continue
            if result:
                return result, provider
        return None, None

ai_generator = AIInsightGenerator()

def analyze_emotions_with_ai(text):
    """Enhanced emotion analysis using AI"""
    try:
        # Basic sentiment analysis
        basic_sentiment = analyze_sentiment(text)
        
        # AI-powered emotion detection
        ai_emotions = {}
        if emotion_classifier:
            emotions = emotion_classifier(text)
            ai_emotions = {emotion['label']: emotion['score'] for emotion in emotions}
        
        # Generate personalized insight
        insight_prompt = f"""
        Analyze this journal entry for emotional patterns and provide a brief, compassionate insight:
        
        "{text}"
        
        Consider the emotional tone, themes, and provide gentle guidance or reflection.
        Keep response under 100 words and be supportive.
        """
        
        ai_insight, provider = ai_generator.generate_insight(insight_prompt)
        
        return {
            **basic_sentiment,
            'ai_emotions': ai_emotions,
            'ai_insight': ai_insight,
            'ai_provider': provider
        }
    except Exception as e:
        logger.error(f"AI emotion analysis error: {e}")
        return analyze_sentiment(text)

def generate_personalized_prompts(user_id, count=5):
    """Generate personalized journaling prompts based on user history"""
    try:
        # Get user's recent reflections
        recent_reflections = list(reflections_collection.find({
            'user_id': user_id
        }).sort('created_at', -1).limit(10))
        
        if not recent_reflections:
            return get_default_prompts(count)
        
        # Analyze patterns
        emotions = [r.get('sentiment', {}).get('emotion', 'neutral') for r in recent_reflections]
        dominant_emotions = list(set(emotions))
        
        # Create context for AI
        context = f"""
        Based on a user's recent emotional patterns showing {', '.join(dominant_emotions)} emotions,
        generate {count} personalized, thoughtful journaling prompts that would help them:
        1. Explore their emotions deeper
        2. Find positive perspectives
        3. Practice gratitude and mindfulness
        4. Build emotional resilience
        
        Make prompts engaging, specific, and supportive. Each prompt should be one sentence.
        """
        
        ai_prompts, provider = ai_generator.generate_insight(context, max_tokens=200)
        
        if ai_prompts:
            # Parse prompts from AI response
            prompts = [line.strip() for line in ai_prompts.split('\n') if line.strip() and not line.strip().isdigit()]
            prompts = [p.lstrip('•-* ') for p in prompts if len(p) > 10]
            
            if len(prompts) >= count:
                return prompts[:count]
        
        # Fallback to default prompts
        return get_default_prompts(count)
        
    except Exception as e:
        logger.error(f"Personalized prompts error: {e}")
        return get_default_prompts(count)

def get_default_prompts(count=5):
    """Get default journaling prompts"""
    prompts = [
        "What am I most grateful for today, and how did it make me feel?",
        "Describe a moment today when I felt completely present and mindful.",
        "What emotion surprised me today, and what might have triggered it?",
        "How did I show kindness to myself or others today?",
        "What challenge did I face today, and what did I learn from it?",
        "What would I tell a good friend who was feeling the way I feel right now?",
        "What are three small things that brought me joy or peace today?",
        "How have I grown emotionally in the past week?",
        "What patterns do I notice in my thoughts and feelings lately?",
        "What intention do I want to set for tomorrow to nurture my well-being?"
    ]
    return random.sample(prompts, min(count, len(prompts)))

def generate_weekly_insight_report(user_id):
    """Generate AI-powered weekly insight report"""
    try:
        # Get week's data
        week_ago = datetime.utcnow() - timedelta(days=7)
        reflections = list(reflections_collection.find({
            'user_id': user_id,
            'created_at': {'$gte': week_ago}
        }).sort('created_at', 1))
        
        if not reflections:
            return None
        
        # Prepare data for AI analysis
        entries_summary = []
        for reflection in reflections:
            summary = {
                'date': reflection['created_at'].strftime('%Y-%m-%d'),
                'mood': reflection.get('mood_rating', 5),
                'emotion': reflection.get('sentiment', {}).get('emotion', 'neutral'),
                'key_themes': reflection.get('content', '')[:200] + "..."
            }
            entries_summary.append(summary)
        
        # Generate comprehensive report
        report_prompt = f"""
        Create a compassionate weekly emotional wellness report based on this user's journal entries:
        
        {json.dumps(entries_summary, indent=2)}
        
        Please provide:
        1. Overall emotional trend for the week
        2. Key patterns or themes observed
        3. Positive developments and strengths shown
        4. Gentle suggestions for continued growth
        5. Encouraging words for the week ahead
        
        Keep the tone supportive, professional, and hopeful. Limit to 300 words.
        """
        
        report, provider = ai_generator.generate_insight(report_prompt, max_tokens=400)
        
        if report:
            # Save insight to database
            insight_data = {
                '_id': str(uuid.uuid4()),
                'user_id': user_id,
                'type': 'weekly_report',
                'content': report,
                'ai_provider': provider,
                'period_start': week_ago,
                'period_end': datetime.utcnow(),
                'created_at': datetime.utcnow()
            }
            ai_insights_collection.insert_one(insight_data)
            
            return report
        
        return None
        
    except Exception as e:
        logger.error(f"Weekly report generation error: {e}")
        return None

def generate_mood_recommendations(current_mood, emotion):
    """Generate AI-powered recommendations based on current mood"""
    try:
        recommendation_prompt = f"""
        A user is currently feeling {emotion} with a mood rating that suggests they are {current_mood}.
        
        Provide 3-4 specific, actionable recommendations to help them:
        1. Process their current emotions healthily
        2. Improve their mood naturally
        3. Practice self-care
        
        Include a mix of immediate actions (5-10 minutes) and longer-term strategies.
        Be specific, practical, and compassionate. Format as bullet points.
        """
        
        recommendations, provider = ai_generator.generate_insight(recommendation_prompt, max_tokens=200)
        
        return recommendations if recommendations else get_default_recommendations(emotion)
        
    except Exception as e:
        logger.error(f"Mood recommendations error: {e}")
        return get_default_recommendations(emotion)

def get_default_recommendations(emotion):
    """Default mood recommendations"""
    recommendations = {
        'negative': [
            "Take 5 deep breaths, focusing on the exhale to release tension",
            "Write down three things you're grateful for right now",
            "Take a short walk outside or look out a window at nature",
            "Listen to calming music or sounds that bring you peace"
        ],
        'positive': [
            "Capture this positive moment by writing about what's going well",
            "Share your positive energy with someone you care about",
            "Take a moment to appreciate and savor this feeling",
            "Consider what actions or thoughts contributed to this mood"
        ],
        'neutral': [
            "Practice a brief mindfulness exercise to connect with the present",
            "Reflect on what you need most right now - rest, activity, or connection",
            "Set a small, achievable goal for the day",
            "Take a few minutes to check in with your body and emotions"
        ]
    }
    return recommendations.get(emotion, recommendations['neutral'])

# Helper Functions (keeping existing ones)
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        user = users_collection.find_one({'_id': session['user_id']})
        if not user or user.get('role') != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def analyze_sentiment(text):
    """Original sentiment analysis (keeping for compatibility)"""
    try:
        # VADER sentiment analysis
        vader_scores = sia.polarity_scores(text)
        
        # TextBlob for additional analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combine scores for more robust analysis
        compound_score = vader_scores['compound']
        
        # Determine emotion category
        if compound_score >= 0.5:
            emotion = 'very_positive'
        elif compound_score >= 0.1:
            emotion = 'positive'
        elif compound_score > -0.1:
            emotion = 'neutral'
        elif compound_score > -0.5:
            emotion = 'negative'
        else:
            emotion = 'very_negative'
        
        return {
            'compound': compound_score,
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'emotion': emotion,
            'polarity': textblob_polarity,
            'subjectivity': textblob_subjectivity
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'emotion': 'neutral',
            'polarity': 0.0,
            'subjectivity': 0.0
        }

def upload_to_s3(file, filename):
    """Upload file to S3 bucket"""
    try:
        s3_client.upload_fileobj(
            file,
            S3_BUCKET,
            filename,
            ExtraArgs={'ACL': 'public-read'}
        )
        return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{filename}"
    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        return None

def send_notification(message, subject="MindfulMuse Notification"):
    """Send notification via SNS"""
    try:
        if SNS_TOPIC_ARN:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject=subject
            )
        return True
    except Exception as e:
        logger.error(f"SNS notification error: {e}")
        return False

def generate_insights(user_id):
    """Enhanced insights generation with AI"""
    try:
        # Get last 30 days of reflections
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        reflections = list(reflections_collection.find({
            'user_id': user_id,
            'created_at': {'$gte': thirty_days_ago}
        }).sort('created_at', 1))
        
        if not reflections:
            return {}
        
        # Calculate trends
        emotions = [r['sentiment']['emotion'] for r in reflections]
        compounds = [r['sentiment']['compound'] for r in reflections]
        
        emotion_counts = defaultdict(int)
        for emotion in emotions:
            emotion_counts[emotion] += 1
        
        avg_sentiment = np.mean(compounds)
        sentiment_trend = "improving" if len(compounds) > 1 and compounds[-1] > compounds[0] else "stable"
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Generate AI-powered personalized insight
        ai_summary_prompt = f"""
        Provide a brief, encouraging summary for a user whose emotional data shows:
        - {len(reflections)} journal entries in the past month
        - Dominant emotion: {dominant_emotion}
        - Average sentiment: {avg_sentiment:.2f} (scale: -1 to 1)
        - Trend: {sentiment_trend}
        
        Give them a supportive insight about their emotional journey and one positive suggestion.
        Keep it under 80 words and be encouraging.
        """
        
        ai_insight, provider = ai_generator.generate_insight(ai_summary_prompt, max_tokens=120)
        
        insights = {
            'total_entries': len(reflections),
            'average_sentiment': avg_sentiment,
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': dict(emotion_counts),
            'trend': sentiment_trend,
            'streak': calculate_streak(user_id),
            'ai_insight': ai_insight,
            'ai_provider': provider,
            'generated_at': datetime.utcnow()
        }
        
        return insights
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        return {}

def calculate_streak(user_id):
    """Calculate current journaling streak"""
    try:
        today = datetime.utcnow().date()
        streak = 0
        current_date = today
        
        while True:
            reflection = reflections_collection.find_one({
                'user_id': user_id,
                'created_at': {
                    '$gte': datetime.combine(current_date, datetime.min.time()),
                    '$lt': datetime.combine(current_date + timedelta(days=1), datetime.min.time())
                }
            })
            
            if reflection:
                streak += 1
                current_date -= timedelta(days=1)
            else:
                break
        
        return streak
    except Exception as e:
        logger.error(f"Streak calculation error: {e}")
        return 0

# Routes (Enhanced with AI)
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            
            # Validation
            if not username or not email or not password:
                return jsonify({'error': 'All fields are required'}), 400
            
            # Check if user exists
            if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
                return jsonify({'error': 'Username or email already exists'}), 400
            
            # Create user
            user_id = str(uuid.uuid4())
            user_data = {
                '_id': user_id,
                'username': username,
                'email': email,
                'password': generate_password_hash(password),
                'role': 'user',
                'created_at': datetime.utcnow(),
                'last_login': None,
                'preferences': {
                    'notification_frequency': 'daily',
                    'theme': 'light',
                    'ai_insights': True,
                    'personalized_prompts': True,
                    'preferred_ai_provider': 'openai'
                }
            }
            
            users_collection.insert_one(user_data)
            
            # Generate personalized welcome message with AI
            welcome_prompt = f"Create a warm, encouraging welcome message for a new user named {username} who just joined a mindfulness journaling app. Keep it under 50 words and focus on their emotional wellness journey."
            welcome_message, _ = ai_generator.generate_insight(welcome_prompt, max_tokens=80)
            
            if not welcome_message:
                welcome_message = f"Welcome to MindfulMuse, {username}! Start your emotional wellness journey today."
            
            # Send welcome notification
            send_notification(welcome_message, "Welcome to MindfulMuse")
            
            return jsonify({'message': 'Registration successful'}), 201
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return jsonify({'error': 'Registration failed'}), 500
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # Find user
            user = users_collection.find_one({'username': username})
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['_id']
                session['username'] = user['username']
                session['role'] = user.get('role', 'user')
                
                # Update last login
                users_collection.update_one(
                    {'_id': user['_id']},
                    {'$set': {'last_login': datetime.utcnow()}}
                )
                
                return jsonify({'message': 'Login successful'}), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'error': 'Login failed'}), 500
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        user_id = session['user_id']
        
        # Get recent reflections
        recent_reflections = list(reflections_collection.find({
            'user_id': user_id
        }).sort('created_at', -1).limit(5))
        
        # Generate enhanced insights with AI
        insights = generate_insights(user_id)
        
        # Get personalized prompts
        personalized_prompts = generate_personalized_prompts(user_id, 3)
        
        # Get mood recommendations if recent reflection exists
        recommendations = []
        if recent_reflections:
            latest_reflection = recent_reflections[0]
            emotion = latest_reflection.get('sentiment', {}).get('emotion', 'neutral')
            mood_rating = latest_reflection.get('mood_rating', 5)
            current_mood = 'positive' if mood_rating >= 7 else 'negative' if mood_rating <= 3 else 'neutral'
            recommendations = generate_mood_recommendations(current_mood, emotion)
        
        return render_template('dashboard.html', 
                             reflections=recent_reflections, 
                             insights=insights,
                             prompts=personalized_prompts,
                             recommendations=recommendations)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard', 'error')
        return render_template('dashboard.html', reflections=[], insights={})

@app.route('/journal', methods=['GET', 'POST'])
@login_required
def journal():
    if request.method == 'POST':
        try:
            data = request.get_json()
            content = data.get('content')
            mood_rating = data.get('mood_rating', 5)
            tags = data.get('tags', [])
            
            if not content:
                return jsonify({'error': 'Content is required'}), 400
            
            # Enhanced sentiment analysis with AI
            sentiment = analyze_emotions_with_ai(content)
            
            # Create reflection
            reflection_data = {
                '_id': str(uuid.uuid4()),
                'user_id': session['user_id'],
                'content': content,
                'mood_rating': mood_rating,
                'tags': tags,
                'sentiment': sentiment,
                'created_at': datetime.utcnow(),
                'word_count': len(content.split())
            }
            
            reflections_collection.insert_one(reflection_data)
            
            # Check for milestones and generate celebratory message
            streak = calculate_streak(session['user_id'])
            milestone_message = None
            
            if streak in [7, 30, 100, 365]:
                celebration_prompt = f"Create a congratulatory message for someone who has achieved a {streak}-day journaling streak. Make it encouraging and celebrate their dedication to emotional wellness. Keep it under 40 words."
                milestone_message, _ = ai_generator.generate_insight(celebration_prompt, max_tokens=60)
                
                if not milestone_message:
                    milestone_message = f"Congratulations! You've reached a {streak}-day journaling streak!"
                
                send_notification(milestone_message, "MindfulMuse Milestone")
            
            # Generate mood-based recommendations
            current_mood = 'positive' if mood_rating >= 7 else 'negative' if mood_rating <= 3 else 'neutral'
            recommendations = generate_mood_recommendations(current_mood, sentiment.get('emotion', 'neutral'))
            
            return jsonify({
                'message': 'Reflection saved successfully',
                'sentiment': sentiment,
                'streak': streak,
                'milestone_message': milestone_message,
                'recommendations': recommendations
            }), 201
            
        except Exception as e:
            logger.error(f"Journal submission error: {e}")
            return jsonify({'error': 'Failed to save reflection'}), 500
    
    return render_template('journal.html')

@app.route('/analytics')
@login_required
def analytics():
    try:
        user_id = session['user_id']
        
        # Get analytics data
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        reflections = list(reflections_collection.find({
            'user_id': user_id,
            'created_at': {'$gte': thirty_days_ago}
        }).sort('created_at', 1))
        
        # Prepare data for charts
        daily_sentiment = {}
        emotion_counts = defaultdict(int)
        
        for reflection in reflections:
            date_key = reflection['created_at'].strftime('%Y-%m-%d')
            daily_sentiment[date_key] = reflection['sentiment']['compound']
            emotion_counts[reflection['sentiment']['emotion']] += 1
        
        # Generate AI-powered analytics insight
        analytics_prompt = f"""
        Analyze this user's emotional data from the past 30 days and provide insights:
        - Total entries: {len(reflections)}
        - Daily sentiment pattern: {len(daily_sentiment)} days with entries
        - Emotion distribution: {dict(emotion_counts)}
        - Average mood: {np.mean([r['mood_rating'] for r in reflections]) if reflections else 0:.1f}/10
        
        Provide a brief, encouraging analysis of their emotional patterns and one actionable suggestion for continued growth.
        Keep it under 100 words and be supportive.
        """
        
        ai_analytics_insight, provider = ai_generator.generate_insight(analytics_prompt, max_tokens=150)
        
        analytics_data = {
            'daily_sentiment': daily_sentiment,
            'emotion_distribution': dict(emotion_counts),
            'total_entries': len(reflections),
            'average_mood': np.mean([r['mood_rating'] for r in reflections]) if reflections else 0,
            'ai_insight': ai_analytics_insight,
            'ai_provider': provider
        }
        
        return render_template('analytics.html', data=analytics_data)
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return render_template('analytics.html', data={
            'total_entries': 0,
            'daily_sentiment': {},
            'emotion_distribution': {},
            'average_mood': 0,
            'ai_insight': '',
            'ai_provider': ''
        })

@app.route('/admin')
@admin_required
def admin_dashboard():
    try:
        # Get system statistics
        total_users = users_collection.count_documents({})
        total_reflections = reflections_collection.count_documents({})
        active_users = users_collection.count_documents({
            'last_login': {'$gte': datetime.utcnow() - timedelta(days=7)}
        })
        
        # Get AI usage statistics
        ai_insights_count = ai_insights_collection.count_documents({})
        ai_provider_stats = list(ai_insights_collection.aggregate([
            {'$group': {'_id': '$ai_provider', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]))
        
        # Get recent activity
        recent_reflections = list(reflections_collection.find({}).sort('created_at', -1).limit(10))
        
        # Generate AI system health report
        system_prompt = f"""
        Generate a brief system health summary for an AI-powered mindfulness app:
        - Total users: {total_users}
        - Active users (7 days): {active_users}
        - Total reflections: {total_reflections}
        - AI insights generated: {ai_insights_count}
        - AI provider usage: {ai_provider_stats}
        
        Provide a concise status report with key metrics and any recommendations for system optimization.
        Keep it under 120 words and focus on actionable insights.
        """
        
        system_health_report, provider = ai_generator.generate_insight(system_prompt, max_tokens=180)
        
        admin_data = {
            'total_users': total_users,
            'total_reflections': total_reflections,
            'active_users': active_users,
            'ai_insights_count': ai_insights_count,
            'ai_provider_stats': ai_provider_stats,
            'recent_reflections': recent_reflections,
            'system_health_report': system_health_report,
            'ai_provider': provider
        }
        
        return render_template('admin.html', data=admin_data)
        
    except Exception as e:
        logger.error(f"Admin dashboard error: {e}")
        return render_template('admin.html', data={})

# New AI-Enhanced API Routes

@app.route('/api/prompts')
@login_required
def get_prompts():
    """Get AI-powered personalized journaling prompts"""
    try:
        count = request.args.get('count', 5, type=int)
        prompts = generate_personalized_prompts(session['user_id'], count)
        return jsonify({'prompts': prompts, 'personalized': True})
    except Exception as e:
        logger.error(f"Prompts API error: {e}")
        default_prompts = get_default_prompts(5)
        return jsonify({'prompts': default_prompts, 'personalized': False})

@app.route('/api/ai-insight', methods=['POST'])
@login_required
def get_ai_insight():
    """Get AI insight for a specific text or question"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        context = data.get('context', 'general')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if context == 'emotion_analysis':
            prompt = f"Analyze the emotional content of this text and provide supportive guidance: '{text}'"
        elif context == 'goal_setting':
            prompt = f"Help create actionable goals based on this reflection: '{text}'"
        elif context == 'gratitude':
            prompt = f"Help identify gratitude opportunities from this experience: '{text}'"
        else:
            prompt = f"Provide thoughtful, supportive insight about: '{text}'"
        
        insight, provider = ai_generator.generate_insight(prompt, max_tokens=200)
        
        if insight:
            return jsonify({
                'insight': insight,
                'provider': provider,
                'context': context
            })
        else:
            return jsonify({'error': 'Unable to generate insight at this time'}), 500
            
    except Exception as e:
        logger.error(f"AI insight API error: {e}")
        return jsonify({'error': 'Insight generation failed'}), 500

@app.route('/api/mood-recommendations')
@login_required
def get_mood_recommendations():
    """Get AI-powered mood-based recommendations"""
    try:
        mood_rating = request.args.get('mood', 5, type=int)
        emotion = request.args.get('emotion', 'neutral')
        
        current_mood = 'positive' if mood_rating >= 7 else 'negative' if mood_rating <= 3 else 'neutral'
        recommendations = generate_mood_recommendations(current_mood, emotion)
        
        return jsonify({
            'recommendations': recommendations,
            'mood_category': current_mood,
            'emotion': emotion
        })
        
    except Exception as e:
        logger.error(f"Mood recommendations API error: {e}")
        return jsonify({'error': 'Unable to generate recommendations'}), 500

@app.route('/api/weekly-report')
@login_required
def get_weekly_report():
    """Generate and retrieve AI-powered weekly insight report"""
    try:
        user_id = session['user_id']
        
        # Check if report already exists for this week
        week_start = datetime.utcnow() - timedelta(days=7)
        existing_report = ai_insights_collection.find_one({
            'user_id': user_id,
            'type': 'weekly_report',
            'created_at': {'$gte': week_start}
        })
        
        if existing_report:
            return jsonify({
                'report': existing_report['content'],
                'generated_at': existing_report['created_at'].isoformat(),
                'provider': existing_report.get('ai_provider', 'unknown')
            })
        
        # Generate new report
        report = generate_weekly_insight_report(user_id)
        
        if report:
            return jsonify({
                'report': report,
                'generated_at': datetime.utcnow().isoformat(),
                'new_report': True
            })
        else:
            return jsonify({'error': 'Insufficient data for weekly report'}), 404
            
    except Exception as e:
        logger.error(f"Weekly report API error: {e}")
        return jsonify({'error': 'Report generation failed'}), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def ai_chat():
    """AI chat interface for emotional support"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get user's recent context for more personalized responses
        recent_reflections = list(reflections_collection.find({
            'user_id': session['user_id']
        }).sort('created_at', -1).limit(3))
        
        context_emotions = []
        if recent_reflections:
            context_emotions = [r.get('sentiment', {}).get('emotion', 'neutral') for r in recent_reflections]
        
        chat_prompt = f"""
        You are a compassionate AI wellness companion. A user is reaching out with: "{message}"
        
        Their recent emotional patterns show: {', '.join(context_emotions) if context_emotions else 'no recent data'}
        
        Respond with empathy, provide gentle guidance, and ask a thoughtful follow-up question if appropriate.
        Keep your response warm, supportive, and under 150 words. Focus on emotional validation and practical wellness tips.
        """
        
        response, provider = ai_generator.generate_insight(chat_prompt, max_tokens=200)
        
        if response:
            # Log the interaction (optional - consider privacy implications)
            chat_log = {
                '_id': str(uuid.uuid4()),
                'user_id': session['user_id'],
                'user_message': message,
                'ai_response': response,
                'ai_provider': provider,
                'created_at': datetime.utcnow()
            }
            # Uncomment to log chat interactions
            # ai_insights_collection.insert_one(chat_log)
            
            return jsonify({
                'response': response,
                'provider': provider,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                'response': "I'm here to listen and support you. Sometimes I might not have the perfect words, but know that your feelings are valid and you're not alone in this journey.",
                'provider': 'fallback',
                'timestamp': datetime.utcnow().isoformat()
            })
            
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return jsonify({'error': 'Chat service temporarily unavailable'}), 500

@app.route('/api/export')
@login_required
def export_data():
    """Export user's reflection data with AI insights"""
    try:
        user_id = session['user_id']
        
        # Get all reflections
        reflections = list(reflections_collection.find({
            'user_id': user_id
        }).sort('created_at', 1))
        
        # Get AI insights
        ai_insights = list(ai_insights_collection.find({
            'user_id': user_id
        }).sort('created_at', 1))
        
        # Convert ObjectId and datetime for JSON serialization
        for reflection in reflections:
            reflection['created_at'] = reflection['created_at'].isoformat()
            
        for insight in ai_insights:
            insight['created_at'] = insight['created_at'].isoformat()
            if 'period_start' in insight:
                insight['period_start'] = insight['period_start'].isoformat()
            if 'period_end' in insight:
                insight['period_end'] = insight['period_end'].isoformat()
        
        export_data = {
            'user_id': user_id,
            'export_date': datetime.utcnow().isoformat(),
            'reflections': reflections,
            'ai_insights': ai_insights,
            'total_reflections': len(reflections),
            'total_ai_insights': len(ai_insights)
        }
        
        return jsonify(export_data)
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': 'Export failed'}), 500

@app.route('/api/ai-settings', methods=['GET', 'POST'])
@login_required
def ai_settings():
    """Manage user's AI preferences"""
    try:
        user_id = session['user_id']
        
        if request.method == 'POST':
            data = request.get_json()
            
            # Update user preferences
            preferences_update = {}
            if 'ai_insights' in data:
                preferences_update['preferences.ai_insights'] = data['ai_insights']
            if 'personalized_prompts' in data:
                preferences_update['preferences.personalized_prompts'] = data['personalized_prompts']
            if 'preferred_ai_provider' in data:
                preferences_update['preferences.preferred_ai_provider'] = data['preferred_ai_provider']
            
            if preferences_update:
                users_collection.update_one(
                    {'_id': user_id},
                    {'$set': preferences_update}
                )
                
            return jsonify({'message': 'AI settings updated successfully'})
        
        else:
            # Get current settings
            user = users_collection.find_one({'_id': user_id})
            preferences = user.get('preferences', {})
            
            return jsonify({
                'ai_insights': preferences.get('ai_insights', True),
                'personalized_prompts': preferences.get('personalized_prompts', True),
                'preferred_ai_provider': preferences.get('preferred_ai_provider', 'openai'),
                'available_providers': ['openai', 'anthropic', 'gemini', 'local']
            })
            
    except Exception as e:
        logger.error(f"AI settings error: {e}")
        return jsonify({'error': 'Settings update failed'}), 500

# Background Tasks (for scheduled AI insights)
@app.route('/api/generate-insights', methods=['POST'])
@admin_required
def trigger_bulk_insights():
    """Admin endpoint to trigger bulk insight generation"""
    try:
        # Get all active users
        active_users = list(users_collection.find({
            'last_login': {'$gte': datetime.utcnow() - timedelta(days=30)},
            'preferences.ai_insights': True
        }))
        
        insights_generated = 0
        
        for user in active_users:
            try:
                # Generate weekly report if it doesn't exist
                week_start = datetime.utcnow() - timedelta(days=7)
                existing_report = ai_insights_collection.find_one({
                    'user_id': user['_id'],
                    'type': 'weekly_report',
                    'created_at': {'$gte': week_start}
                })
                
                if not existing_report:
                    report = generate_weekly_insight_report(user['_id'])
                    if report:
                        insights_generated += 1
                        
            except Exception as e:
                logger.error(f"Bulk insight generation error for user {user['_id']}: {e}")
                continue
        
        return jsonify({
            'message': f'Generated insights for {insights_generated} users',
            'total_processed': len(active_users)
        })
        
    except Exception as e:
        logger.error(f"Bulk insights trigger error: {e}")
        return jsonify({'error': 'Bulk insight generation failed'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
