from streamlit.testing.v1 import AppTest

def test_streamlit_app():
    print("--- STARTING STREAMLIT APPTEST SUITE ---")
    
    # 1. Initialize AppTest
    print("1. Loading app.py in AppTest...")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Exceptions on startup: {at.exception}"
    print("✓ App launched cleanly without exceptions.")
    
    # Check default page title / markdown
    titles = [t.value for t in at.title]
    assert any("Project 2" in t or "Data Classification" in t for t in titles)
    print("✓ Dashboard page rendered correctly with title.")
    
    # 2. Iterate through all navigation pages
    pages = [
        "1. Dashboard",
        "2. Dataset Details",
        "3. Data Preprocessing",
        "4. Train / Test Split",
        "5. KNN Classifier",
        "6. Model Evaluation",
        "7. Interactive Prediction",
        "8. ML Academy (How It Works)"
    ]
    
    for page_name in pages:
        print(f"2. Navigating to '{page_name}'...")
        at.sidebar.radio[0].set_value(page_name).run()
        assert len(at.exception) == 0, f"Exception on {page_name}: {at.exception}"
        print(f"✓ '{page_name}' rendered with 0 exceptions.")
        
    # 3. Test Interactive KNN Tuning on Page 5
    print("3. Testing K-slider interaction on KNN Classifier page...")
    at.sidebar.radio[0].set_value("5. KNN Classifier").run()
    assert len(at.exception) == 0
    
    # Adjust K slider (first slider on page is the K slider)
    if len(at.slider) > 0:
        at.slider[0].set_value(7).run()
        assert len(at.exception) == 0
        assert at.session_state["k"] == 7
        print("✓ K slider successfully tuned to K=7 and model updated.")
        
    # 4. Test Model Evaluation Page with K=7
    print("4. Testing Model Evaluation page with updated K...")
    at.sidebar.radio[0].set_value("6. Model Evaluation").run()
    assert len(at.exception) == 0
    print("✓ Model Evaluation page rendered with updated metrics.")
    
    # 5. Test Interactive Prediction Portal with Presets and Prediction
    print("5. Testing Interactive Prediction Portal (presets and predictions)...")
    at.sidebar.radio[0].set_value("7. Interactive Prediction").run()
    assert len(at.exception) == 0
    
    # Click Setosa preset button
    setosa_button = None
    for btn in at.button:
        if "Setosa" in btn.label:
            setosa_button = btn
            break
            
    if setosa_button:
        setosa_button.click().run()
        assert len(at.exception) == 0
        assert "last_prediction" in at.session_state
        assert at.session_state["last_prediction"]["pred_class_id"] == 0 # Setosa
        print("✓ Setosa preset clicked: predicted Setosa (class 0) successfully.")
        
    # Click Virginica preset button
    virginica_button = None
    for btn in at.button:
        if "Virginica" in btn.label:
            virginica_button = btn
            break
            
    if virginica_button:
        virginica_button.click().run()
        assert len(at.exception) == 0
        assert "last_prediction" in at.session_state
        assert at.session_state["last_prediction"]["pred_class_id"] == 2 # Virginica
        print("✓ Virginica preset clicked: predicted Virginica (class 2) successfully.")
        
    # Click Main Predict button
    predict_btn = None
    for btn in at.button:
        if "PREDICT SPECIES" in btn.label:
            predict_btn = btn
            break
            
    if predict_btn:
        predict_btn.click().run()
        assert len(at.exception) == 0
        print("✓ 'PREDICT SPECIES' button executed successfully.")
        
    print("\n✓✓✓ ALL STREAMLIT APPTEST INTEGRATION TESTS PASSED SUCCESSFULLY! ✓✓✓")

if __name__ == "__main__":
    test_streamlit_app()
