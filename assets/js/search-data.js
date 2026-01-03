// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A collection of projects showcasing my work in machine learning, robotics, and practical software systems.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-ai-can-learn-from-their-dreams-world-models",
        
          title: 'AI Can Learn From Their Dreams: World Models <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "World models as imagination engines for intelligent agents.",
        section: "Posts",
        handler: () => {
          
            window.open("https://medium.com/@humansforai/ai-can-learn-from-their-dreams-world-models-3018fb21602b", "_blank");
          
        },
      },{id: "post-ai-amp-ml-in-autonomous-driving",
        
          title: 'AI &amp; ML in Autonomous Driving <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "How modern learning-based systems power autonomy and driving stacks.",
        section: "Posts",
        handler: () => {
          
            window.open("https://medium.com/@humansforai/ai-ml-in-autonomous-driving-3fbb992dcfc4", "_blank");
          
        },
      },{id: "post-self-supervised-learning-what-is-it",
        
          title: 'Self-Supervised Learning: What is it? <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "A practical overview of self-supervised learning and why it works.",
        section: "Posts",
        handler: () => {
          
            window.open("https://medium.com/@humansforai/self-supervised-learning-what-is-it-5d00fa1c8b8e", "_blank");
          
        },
      },{id: "post-contrastive-deep-explanations",
        
          title: "Contrastive Deep Explanations",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/Contrastive-Deep-Explanations/";
          
        },
      },{id: "post-bootstrap-your-own-latent-self-supervised-learning-without-contrastive-learning",
        
          title: "Bootstrap Your Own Latent: Self-Supervised Learning Without Contrastive Learning",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/bootstrap-your-own-latent/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "projects-mydreamerv2",
          title: 'MyDreamerv2',
          description: "A reimplementation and extension of DreamerV2 with exploration via Plan2Explore and selected improvements from DreamerV3",
          section: "Projects",handler: () => {
              window.location.href = "/projects/MyDreamerv2/";
            },},{id: "projects-salesense",
          title: 'SaleSense',
          description: "A tool that scores resale listings and generates clear, higher-converting product descriptions.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/SaleSense/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%61%6D%61%76%69%7A%63%61%70%65%64%72%6F@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/PedroTajia", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/pedro-tajia-amavizca-282913280", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
