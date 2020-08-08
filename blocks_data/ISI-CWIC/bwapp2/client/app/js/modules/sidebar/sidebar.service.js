(function() {
    'use strict';

    angular
        .module('app.sidebar')
        .service('SidebarLoader', SidebarLoader);

    SidebarLoader.$inject = ['$meteor'];
    function SidebarLoader($meteor) {
        this.getMenu = getMenu;

        ////////////////

        function getMenu(onReady, onError) {

          onError = onError || function() { alert('Failure loading menu'); };

          $meteor.call('sidebar').then(
            onReady,
            onError
          );

        }
    }
})();