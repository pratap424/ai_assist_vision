import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_NAME = "Salesforce/blip-vqa-base"  # Smaller efficient model for edge devices

class VQAProcessor:
    def __init__(self):
        """Initialize the Visual Question Answering model"""
        print("📚 Loading VQA model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BLIP VQA model
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(self.device)
        
        # Optimize the model for inference
        self.model.eval()  # Set to evaluation mode
        if self.device.type == "cuda":
            # Use mixed precision for faster inference on GPU
            self.model = self.model.half()
        
        print(f"✅ VQA model loaded on {self.device}")
        
    def answer_question(self, image_path, question):
        """
        Answer a natural language question about an image
        
        Args:
            image_path (str): Path to the image
            question (str): Question about the image
            
        Returns:
            str: Answer to the question
        """
        print(f"❓ Processing question: '{question}'")
        
        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess the inputs
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"💡 Answer: '{answer}'")
        return answer
    
    def parse_and_enhance_query(self, query, image_data=None):
        """
        Process the query to handle specific types of questions better
        
        Args:
            query (str): The original query
            image_data (dict): Data from other modules about the image
            
        Returns:
            str: Enhanced query or original if no enhancement needed
        """
        # Standardize query
        query = query.lower().strip()
        
        # Dictionary of query patterns and their enhancements
        query_patterns = {
            "how many": query,  # Keep as is
            "what color": query,  # Keep as is
            "where is": query,  # Keep as is
            "is there": query,  # Keep as is
            "what is in the background": "Describe the background of this image",
            "what is on the left": "What objects are on the left side of the image",
            "what is on the right": "What objects are on the right side of the image",
            "who is": query,  # Keep as is
        }
        
        # Check if query matches any patterns
        for pattern, enhanced_query in query_patterns.items():
            if pattern in query:
                return enhanced_query
                
        # Default fallback - if no pattern matched
        return query

    def answer_query_with_context(self, image_path, query, image_data=None):
        """
        Answer a query using both VQA and existing image analysis data
        
        Args:
            image_path (str): Path to the image
            query (str): User's query about the image
            image_data (dict): Data from other modules (objects, regions, etc.)
            
        Returns:
            str: Natural language answer
        """
        # First try answering directly with VQA
        enhanced_query = self.parse_and_enhance_query(query, image_data)
        answer = self.answer_question(image_path, enhanced_query)
        
        # If we have image data and a simple/short answer, try to enhance it
        if image_data and len(answer.split()) < 4:
            query_lower = query.lower()
            
            # Handle count questions by using detector data
            if "how many" in query_lower:
                obj_type = query_lower.replace("how many", "").strip().rstrip("?").strip()
                if "object" in obj_type or "thing" in obj_type:
                    count = sum(image_data.get("objects_detected", {}).values())
                    return f"There are {count} objects in the image."
                
                # Check for specific object type
                for obj, count in image_data.get("objects_detected", {}).items():
                    if obj in obj_type or obj_type in obj:
                        return f"There are {count} {obj}(s) in the image."
            
            # Handle location questions
            elif "where" in query_lower:
                for obj in image_data.get("objects_detected", {}):
                    if obj in query_lower:
                        # Find the region with this object
                        for region in image_data.get("regions", []):
                            if obj in region:
                                return f"The {obj} is {region}."
            
            # Try to handle "what's on the left/right/etc."
            directions = ["left", "right", "center", "top", "bottom"]
            for direction in directions:
                if direction in query_lower:
                    for region in image_data.get("regions", []):
                        if direction in region:
                            return f"On the {direction}, there is {region}."
        
        return answer