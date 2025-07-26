#!/usr/bin/env python3
"""
ECG Analysis with Google Gemini - Working Demo
==============================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def create_sample_ecg():
    """Generate realistic ECG data"""
    print("📊 Creating sample ECG...")
    
    # 5-second ECG at 500 Hz
    time = np.linspace(0, 5, 2500)
    heart_rate = 72  # BPM
    
    # Generate beats
    ecg = np.zeros_like(time)
    beat_times = np.arange(0.5, 5, 60/heart_rate)
    
    for beat in beat_times:
        # QRS complex
        qrs_mask = (time >= beat-0.04) & (time <= beat+0.04)
        ecg[qrs_mask] += 1.2 * np.exp(-((time[qrs_mask] - beat) / 0.015)**2)
        
        # T wave
        t_mask = (time >= beat+0.2) & (time <= beat+0.35)
        ecg[t_mask] += 0.3 * np.exp(-((time[t_mask] - beat - 0.275) / 0.05)**2)
    
    # Add noise
    ecg += 0.05 * np.random.normal(0, 1, len(time))
    
    print(f"✅ Generated ECG: {len(beat_times)} beats, HR: {heart_rate} BPM")
    return time, ecg, heart_rate, len(beat_times)

def analyze_ecg_with_gemini(heart_rate, num_beats):
    """Quick ECG analysis with Gemini"""
    print("🤖 Analyzing with Gemini...")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
ECG Analysis Request:
- Heart Rate: {heart_rate} BPM
- Beats recorded: {num_beats} in 5 seconds
- Rhythm: Regular

As a cardiologist, provide:
1. Is this normal? (Yes/No)
2. Key finding (1 sentence)
3. Recommendation (1 sentence)

Keep response under 50 words.
"""
        
        response = model.generate_content(prompt)
        
        print("✅ Analysis complete!")
        print("\n🏥 MEDICAL ANALYSIS:")
        print("-" * 30)
        print(response.text)
        print("-" * 30)
        
        return response.text
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def create_ecg_plot(time, ecg, heart_rate):
    """Create ECG visualization"""
    print("📈 Creating ECG plot...")
    
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(time, ecg, 'b-', linewidth=1.5, label='ECG Signal')
        plt.title(f'ECG Recording - Heart Rate: {heart_rate} BPM', fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save
        plt.savefig('ecg_analysis.png', dpi=200, bbox_inches='tight')
        print("✅ Plot saved as 'ecg_analysis.png'")
        
        try:
            plt.show()
        except:
            print("📊 Plot created successfully")
            
        return True
        
    except Exception as e:
        print(f"⚠️ Plot error: {e}")
        return False

def main():
    """Run ECG analysis demo"""
    print("🏥 ECG Analysis with Google Gemini")
    print("="*40)
    
    # Generate ECG
    time, ecg, heart_rate, num_beats = create_sample_ecg()
    
    # Analyze with Gemini
    analysis = analyze_ecg_with_gemini(heart_rate, num_beats)
    
    # Create visualization
    create_ecg_plot(time, ecg, heart_rate)
    
    print("\n🎉 Demo completed!")
    print("\n💡 What you can do now:")
    print("• Use real ECG data files")
    print("• Build patient reports")
    print("• Create medical apps")
    print("• Analyze heart conditions")
    
    return True

if __name__ == "__main__":
    main()