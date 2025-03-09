# Automatic Taxonomy Generator

## User Experience

### Tag Generation
- Ask the user how many tags they want
- Then parse the content to establish the tags based on how often they appear semantically in the content
- Be sure to read all the content
- Save the tags to the output folder as `tags.json`  
    - Example: 
        ```
        {
            "tagName": "apple",
            "frequency": "12"
        }        
        ```

