(venv) annex@AmmarKhan Roadshow3DLUT % python sample_manager.py
📁 Created sample directory structure
🎬 ROADSHOW SAMPLE MANAGER
==================================================
🎥 Downloading RED camera samples...
  ❌ Could not download epic_5k_sample.r3d
  ❌ Could not download weapon_8k_sample.r3d
🎨 Creating synthetic test samples...
Traceback (most recent call last):
  File "/Users/annex/Desktop/Roadshow3DLUT/sample_manager.py", line 263, in <module>
    manager.run_full_setup()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/annex/Desktop/Roadshow3DLUT/sample_manager.py", line 249, in run_full_setup
    self.create_test_samples()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/annex/Desktop/Roadshow3DLUT/sample_manager.py", line 64, in create_test_samples
    iphone_sample = self.create_iphone_style_image()
  File "/Users/annex/Desktop/Roadshow3DLUT/sample_manager.py", line 81, in create_iphone_style_image
    img[y, :] = [255, color + 100, 180]  # iPhone blue tint
    ~~~^^^^^^
OverflowError: Python integer 256 out of bounds for uint8
(venv) annex@AmmarKhan Roadshow3DLUT % python simple_sample_tester.py

📁 Created test directories
🎬 ROADSHOW SIMPLE SAMPLE TESTER
==================================================
🎨 Creating test images...
Traceback (most recent call last):
  File "/Users/annex/Desktop/Roadshow3DLUT/simple_sample_tester.py", line 304, in <module>
    tester.run_full_test()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/annex/Desktop/Roadshow3DLUT/simple_sample_tester.py", line 290, in run_full_test
    test_images = self.create_test_images()
  File "/Users/annex/Desktop/Roadshow3DLUT/simple_sample_tester.py", line 45, in create_test_images
    landscape = self.create_landscape_test()
  File "/Users/annex/Desktop/Roadshow3DLUT/simple_sample_tester.py", line 95, in create_landscape_test
    img[y, :] = [255, 200 + blue_amount//3, blue_amount]
    ~~~^^^^^^
OverflowError: Python integer 285 out of bounds for uint8
(venv) annex@AmmarKhan Roadshow3DLUT % 
