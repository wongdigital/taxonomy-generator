# Automatic Taxonomy Generator

## User Experience

### Category Generation
If the input folder does not contain `categories.json` then enter into category generation mode.

- Ask the user how many general categories they want
- Then parse the content to establish the general categories based on how often they appear semantically
- Standardize on that set
- Save the set of categories and their volume to the output folder as `categories.json`
    - Example: 
        ```
        {
            "categoryName": "apple",
            "frequency": "5"
        }        
        ```

### Tag Generation
If the input folder contains `categories.json` then enter into tag generation mode.

- Ask the user how many tags they want
- Then parse the content to establish the tags based on the categories and how often they appear semantically
- Save the tags to the output folder as `taxonomy.json`  
    - Example: 
        ```
        {
            "categoryName": "apple",
            "tags": ["iPhone", "Mac", "macOS", "AppleTV"]
        }        
        ```
