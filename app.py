import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import random
import sys
from datetime import datetime
import plotly.express as px
from typing import Dict, Any
import pandas as pd

# Configure page with waste management theme
st.set_page_config(
    page_title="Waste.ai",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with improved colors and fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Base styles */
    .main {
        background-color: #F8FAF9;
        padding: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        color: #1A3D37;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #34A853;
        color: white;
        width: 100%;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2D8E47;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 168, 83, 0.2);
    }
    
    /* Cards and Containers */
    .dashboard-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(26, 61, 55, 0.08);
        transition: all 0.3s ease;
        border: 1px solid rgba(52, 168, 83, 0.1);
    }
    .dashboard-card:hover {
        box-shadow: 0 8px 24px rgba(26, 61, 55, 0.12);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .metrics-card {
        background: linear-gradient(135deg, #34A853 0%, #2D8E47 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .metrics-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metrics-card .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-good { background-color: #34A853; }
    .status-warning { background-color: #FBBC05; }
    .status-alert { background-color: #EA4335; }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1A3D37;
        color: white;
    }
    
    /* Forms */
    div[data-testid="stForm"] {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(26, 61, 55, 0.08);
        border: 1px solid rgba(52, 168, 83, 0.1);
    }
    
    /* Charts and Graphs */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(26, 61, 55, 0.08);
        border: 1px solid rgba(52, 168, 83, 0.1);
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background-color: #34A853;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(26, 61, 55, 0.08);
    }
    
    /* Alerts and Info boxes */
    .success-alert {
        background-color: rgba(52, 168, 83, 0.1);
        color: #2D8E47;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #34A853;
    }
    .info-alert {
        background-color: rgba(66, 133, 244, 0.1);
        color: #4285F4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4285F4;
    }
    .warning-alert {
        background-color: rgba(251, 188, 5, 0.1);
        color: #F9AB00;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FBBC05;
    }
            
    /* Deadline notifications */
            
    </style>
""", unsafe_allow_html=True)

# Configuration setup remains the same
CONFIG_FILE = 'Gemini_config.json'
MODEL_NAME = 'gemini-1.5-flash-exp-0827'

def load_config(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"Error: Config file not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        st.error(f"Error: Invalid JSON in config file {file_path}")
        sys.exit(1)

def setup_gemini(config: Dict[str, Any]):
    try:
        genai.configure(api_key=st.secrets["API_KEY"])
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        system_prompt = """You are a Waste Management AI Assistant specialized in:
        1. Identifying and categorizing waste materials
        2. Providing recycling and disposal guidelines
        3. Suggesting waste reduction strategies
        4. Explaining environmental impact
        5. Recommending sustainable alternatives
        Please provide specific, actionable advice while maintaining environmental consciousness."""
        
        return genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=system_prompt,
            generation_config=config['generation_config'],
            safety_settings=safety_settings,
        )
    except Exception as e:
        st.error(f"Error setting up Gemini: {str(e)}")
        sys.exit(1)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        config = load_config(CONFIG_FILE)
        st.session_state.model = setup_gemini(config)
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "waste_log" not in st.session_state:
        st.session_state.waste_log = []
    if "collection_schedules" not in st.session_state:
        st.session_state.collection_schedules = []
    if "daily_facts" not in st.session_state:
        st.session_state.daily_facts = []
    if "daily_tip" not in st.session_state:
        st.session_state.daily_tip = ""

def navigation():
    st.sidebar.title("Navigation")
    pages = {
        "home": "🏠 Dashboard",
        "chat": "💬 AI Assistant",
        "tracker": "📊 Waste Tracker",
        "schedule": "📅 Collection Schedule",
    }
    
    st.sidebar.markdown("---")
    for page_id, page_name in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_id}"):
            st.session_state.page = page_id
            st.rerun()

def get_random_fact():
    """Returns a random fact from the FACTS list."""
    # Define a list of facts
    FACTS = [
        "Recycling one aluminum can saves enough energy to run a TV for three hours.",
        "Only 9% of all plastic waste ever produced has been recycled.",
        "Composting food waste can reduce methane emissions from landfills.",
        "Glass is 100% recyclable and can be recycled endlessly without loss of quality.",
        "Around 1 million plastic bottles are bought every minute worldwide.",
        "Recycling one ton of paper can save 17 trees and 7,000 gallons of water.",
        "The decomposition of plastic can take up to 1,000 years in landfills.",
        "Every year, 8 million metric tons of plastic enter the ocean.",
        "Reducing food waste is one of the top ways to combat climate change.",
        "Americans throw away 25% more trash during Thanksgiving to New Year's holiday period.",
        "E-waste represents 2% of trash in landfills but 70% of toxic waste.",
        "Every ton of recycled steel saves 2,500 pounds of iron ore and 1,400 pounds of coal.",
        "Up to 60% of the waste in the average trash bin could be recycled.",
        "Landfills are the third-largest source of methane emissions globally.",
        "Plastic straws cannot be recycled and often end up harming marine life.",
        "Recycling one ton of plastic saves about 2,000 gallons of gasoline.",
        "By 2050, it’s estimated there will be more plastic in the ocean than fish (by weight).",
        "Recycling one glass bottle saves enough energy to power a computer for 30 minutes.",
        "Nearly 1/3 of the food produced globally goes to waste every year.",
        "Upcycling—reusing items creatively—can extend the lifecycle of products and reduce waste.",
        "If every American recycled just one-tenth of their newspapers, 25 million trees could be saved each year.",
        "Textiles like clothing can take up to 200 years to decompose in landfills.",
        "Battery recycling prevents heavy metals from contaminating soil and water."
    ]
    return random.choice(FACTS)

def render_home():
    st.title("♻️ Smart Waste Management Dashboard")
    
    # Overview Section with Enhanced Metrics
    st.markdown("### 📊 Overview")
    metrics_cols = st.columns(4)

    with metrics_cols[0]:
        st.markdown("""
            <div class="metrics-card">
                <div class="metric-value">📦 {}kg</div>
                <div class="metric-label">Total Waste Collected</div>
            </div>
        """.format(sum(item['weight'] for item in st.session_state.get('waste_log', []))),
        unsafe_allow_html=True)

    with metrics_cols[1]:
        completed_goals = len([goal for goal in st.session_state.get('goals', []) if goal['status'] == "Completed"])
        st.markdown(f"""
            <div class="metrics-card">
                <div class="metric-value">🎯 {completed_goals}</div>
                <div class="metric-label">Goals Completed</div>
            </div>
        """, unsafe_allow_html=True)
    
     # Add Random Fact Section
    st.markdown("### 🌟 Did You Know?")
    random_fact = get_random_fact()  # Fetch a new random fact
    st.markdown(f"""
        <div class="dashboard-card">
            <h4 style="color: #1A3D37;">📚 Waste Management Fact</h4>
            <div style="margin: 1rem 0;">
                <div class="info-alert" style="color: #2C5282;">
                    🌍 {random_fact}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with metrics_cols[2]:
        total_schedules = len(st.session_state.get('collection_schedules', []))
        completed_schedules = len(st.session_state.get('completed_schedules', set()))
        pending_schedules = total_schedules - completed_schedules
        st.markdown(f"""
            <div class="metrics-card">
                <div class="metric-value">📅 {pending_schedules}</div>
                <div class="metric-label">Pending Schedules</div>
            </div>
        """, unsafe_allow_html=True)

    with metrics_cols[3]:
        total_recyclables = sum(
            item['weight'] for item in st.session_state.get('waste_log', []) if item['type'] == "Recyclables"
        )
        st.markdown(f"""
            <div class="metrics-card">
                <div class="metric-value">♻️ {total_recyclables}kg</div>
                <div class="metric-label">Recyclables Collected</div>
            </div>
        """, unsafe_allow_html=True)

    # Add interactive chart
    st.markdown("### 📈 Waste Analysis")
    render_waste_chart()  # Ensure this function is defined and works

    # Upcoming Collections (Non-completed schedules only)
    st.markdown("### 📅 Upcoming Collections")
    non_completed_schedules = [
        schedule for schedule in st.session_state.get('collection_schedules', [])
        if schedule['id'] not in st.session_state.get('completed_schedules', set())
    ]

    if non_completed_schedules:
        with st.container():
            for schedule in non_completed_schedules:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                        <div class="info-alert">
                            <strong>{schedule['type']}</strong><br>
                            {schedule['day']} at {schedule['time']} ({schedule['frequency']})
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("✓ Complete", key=f"complete_{schedule['id']}"):
                        mark_schedule_complete(schedule['id'])
    else:
        st.info("No upcoming collections scheduled. Great job staying on top of your tasks!")

    # Recent Activity
    st.markdown("### 🌟 Recent Activities")
    if st.session_state.get('waste_log'):
        recent_logs = sorted(st.session_state['waste_log'], key=lambda x: x['date'], reverse=True)[:3]
        for log in recent_logs:
            st.markdown(f"""
                <div class="success-alert">
                    <strong>{log['date'].strftime('%Y-%m-%d')}</strong><br>
                    {log['type']}: {log['weight']}kg
                    {f"<br>{log['notes']}" if log['notes'] else ""}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent activities logged.")

    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    action_cols = st.columns(4)

    with action_cols[0]:
        if st.button("📝 Log New Waste"):
            st.session_state.page = "tracker"
            st.rerun()

    with action_cols[1]:
        if st.button("📅 View Schedule"):
            st.session_state.page = "schedule"
            st.rerun()

    with action_cols[2]:
        if st.button("🎯 Set New Goal"):
            st.session_state.page = "goals"
            st.rerun()

    with action_cols[3]:
        if st.button("💬 Get AI Help"):
            st.session_state.page = "chat"
            st.rerun()

def render_chat():
    st.title("💬 AI Waste Management Assistant")
    
    # Display chat messages with improved styling
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Enhanced chat input
    if prompt := st.chat_input("Ask about waste management..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.model.generate_content(prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

def render_tracker():
    st.title("📊 Waste Tracker")
    
    # Add new waste entry with improved form styling
    with st.form("waste_entry", clear_on_submit=True):
        st.markdown("### 📝 Log New Waste")
        waste_type = st.selectbox("Waste Type", ["Recyclables", "Organic", "General", "Hazardous", "E-Waste"])
        
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight (kg)", min_value=0.1, step=0.1)
        with col2:
            date = st.date_input("Date")
            
        notes = st.text_area("Notes", placeholder="Add any additional details...")
        
        submit_col1, submit_col2 = st.columns([1, 4])
        with submit_col1:
            submitted = st.form_submit_button("Log Waste")
            
    if submitted:
        st.session_state.waste_log.append({
            "date": date,
            "type": waste_type,
            "weight": weight,
            "notes": notes
        })
        st.success("✅ Waste logged successfully!")
    
    # Display waste log with improved visualization
    if st.session_state.waste_log:
        st.markdown("### 📈 Waste Analysis")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(st.session_state.waste_log)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Waste", f"{df['weight'].sum():.1f} kg")
        with col2:
            st.metric("Most Common Type", df['type'].mode()[0])
        with col3:
            st.metric("Number of Entries", len(df))
        
        # Display detailed log
        st.markdown("### 📋 Detailed Log")
        st.dataframe(
            df.sort_values(by='date', ascending=False),
            use_container_width=True,
            hide_index=True
        )

def render_waste_chart():
    if st.session_state.waste_log:
        df = pd.DataFrame(st.session_state.waste_log)
        fig = px.pie(
            df,
            names='type',
            values='weight',
            title='Waste Composition',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No waste data available. Start logging waste to see insights!")

def delete_schedule(schedule_id):
    if 'collection_schedules' not in st.session_state:
        st.session_state.collection_schedules = []
    st.session_state.collection_schedules = [
        schedule for schedule in st.session_state.collection_schedules
        if schedule['id'] != schedule_id
    ]
    st.rerun()  # Trigger a rerun after deletion

def mark_schedule_complete(schedule_id):
    if 'completed_schedules' not in st.session_state:
        st.session_state.completed_schedules = set()
    st.session_state.completed_schedules.add(schedule_id)
    st.rerun()  # Update the UI after marking complete


def render_schedule():
    st.title("📅 Collection Schedule")

    # Add new collection schedule form
    with st.form("add_schedule"):
        st.markdown("### 📝 Add Collection Schedule")
        
        col1, col2 = st.columns(2)
        with col1:
            waste_type = st.selectbox("Waste Type", ["General", "Recyclables", "Organic", "Hazardous"])
            collection_day = st.selectbox("Collection Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        with col2:
            frequency = st.selectbox("Frequency", ["Weekly", "Bi-weekly", "Monthly"])
            time = st.time_input("Collection Time")

        notes = st.text_area("Additional Notes", placeholder="Add any special instructions...")

        if st.form_submit_button("Add Schedule"):
            if 'collection_schedules' not in st.session_state:
                st.session_state.collection_schedules = []
            new_schedule = {
                "id": len(st.session_state.collection_schedules) + 1,
                "type": waste_type,
                "day": collection_day,
                "time": time.strftime("%I:%M %p"),
                "frequency": frequency,
                "notes": notes
            }
            st.session_state.collection_schedules.append(new_schedule)
            st.success(f"✅ Added {waste_type} collection for {collection_day}s at {time.strftime('%I:%M %p')} ({frequency})")
            st.rerun()

    # Separate Completed and Non-Completed Schedules
    completed_schedules = [
        schedule for schedule in st.session_state.collection_schedules
        if schedule['id'] in st.session_state.get('completed_schedules', set())
    ]
    non_completed_schedules = [
        schedule for schedule in st.session_state.collection_schedules
        if schedule['id'] not in st.session_state.get('completed_schedules', set())
    ]

    # Display Non-Completed Schedules
    st.markdown("### 📋 Non-Completed Schedules")
    if non_completed_schedules:
        for schedule in non_completed_schedules:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                    <div class="info-alert" style="margin-bottom: 0.5rem;">
                        <strong>{schedule['type']}</strong><br>
                        {schedule['day']} at {schedule['time']} ({schedule['frequency']})
                        {f"<br><small>{schedule['notes']}</small>" if schedule.get('notes') else ""}
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("✓ Complete", key=f"complete_schedule_{schedule['id']}"):
                    mark_schedule_complete(schedule['id'])

            with col3:
                if st.button("🗑️ Delete", key=f"delete_schedule_{schedule['id']}"):
                    delete_schedule(schedule['id'])
    else:
        st.info("No non-completed schedules. Great job staying on top of your collections!")

    # Display Completed Schedules
    st.markdown("### ✅ Completed Schedules")
    if completed_schedules:
        for schedule in completed_schedules:
            col1, col3 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                    <div class="success-alert" style="margin-bottom: 0.5rem;">
                        <strong>{schedule['type']}</strong><br>
                        {schedule['day']} at {schedule['time']} ({schedule['frequency']})
                        <br><small>Marked as completed</small>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_completed_{schedule['id']}"):
                    delete_schedule(schedule['id'])
    else:
        st.info("No completed schedules yet.")



def main():
    initialize_session_state()
    navigation()
    
    # Render back button (except on home page)
    if st.session_state.page != "home":
        if st.button("← Back to Dashboard", key="back", use_container_width=False):
            st.session_state.page = "home"
            st.rerun()
    
    # Page routing
    pages = {
        "home": render_home,
        "chat": render_chat,
        "tracker": render_tracker,
        "schedule": render_schedule,
    }
    
    # Render current page
    current_page = pages.get(st.session_state.page, render_home)
    current_page()

if __name__ == "__main__":
    main()
