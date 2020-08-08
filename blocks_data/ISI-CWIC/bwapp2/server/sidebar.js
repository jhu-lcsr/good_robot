
Meteor.methods({
  sidebar: function(){

    var menu = [
      {
        "text": "Menu Heading",
        "heading": "true",
        "translate": "sidebar.heading.HEADER"
      },
      {
        "text": "Single View",
        "sref": "app.singleview",
        "icon": "fa fa-file-o",
        "translate": "sidebar.nav.SINGLEVIEW"
      },
      {
        "text": "Menu",
        "sref": "#",
        "icon": "icon-folder",
        "submenu": [
          { "text": "Sub Menu", 
            "sref": "app.submenu", 
            "translate": "sidebar.nav.menu.SUBMENU" 
          }    
        ],
        "translate": "sidebar.nav.menu.MENU"
      }
    ];
    
    return menu;
  }

});
