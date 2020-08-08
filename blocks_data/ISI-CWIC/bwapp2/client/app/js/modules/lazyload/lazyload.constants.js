(function() {
    'use strict';

    angular
        .module('app.lazyload')
        .constant('APP_REQUIRES', {
          // jQuery based and standalone scripts
          scripts: {
            'modernizr':          ['vendor/modernizr/modernizr.js'],
            'icons':              ['vendor/fontawesome/css/font-awesome.min.css',
                                   'vendor/simple-line-icons/css/simple-line-icons.css'],
            /*'glyphiconspro':      [
              'vendor/glyphicons-pro/glyphicons_halflings/web/html_css/css/halflings.css',
              'vendor/glyphicons-pro/glyphicons/web/html_css/css/glyphicons.css',
              'vendor/glyphicons-pro/glyphicons_social/web/html_css/css/social.css',
              'vendor/glyphicons-pro/glyphicons_filetypes/web/html_css/css/filetypes.css'
            ],*/
            'babylonjs':          [
              'vendor/babylonjs/babylon.2.2.js',
              'vendor/babylonjs/Oimo.js',
              'vendor/handjs/hand.js'],
            'clipboard':          [
              'vendor/clipboard/dist/clipboard.min.js'],
            'levenshtein':          [
              'vendor/fast-levenshtein/levenshtein.min.js']
          },
          // Angular based script (use the right module name)
          modules: [
            // {name: 'toaster', files: ['vendor/angularjs-toaster/toaster.js', 'vendor/angularjs-toaster/toaster.css']}
            {name: 'toaster',                 files: [
              'vendor/angularjs-toaster/toaster.js',
              'vendor/angularjs-toaster/toaster.css'
            ]},
            {name: 'ngMd5',                   files: [
              'vendor/angular-md5/angular-md5.min.js'
            ]},
            {name: 'ngDialog',                files: [
              'vendor/ngDialog/css/ngDialog.min.css',
              'vendor/ngDialog/css/ngDialog-theme-default.min.css',
              'vendor/ngDialog/js/ngDialog.min.js'
            ]},
            {name: 'ngTable',                 files: [
              'vendor/ng-table/dist/ng-table.css',
              'vendor/ng-table/dist/ng-table.js'
            ]},
            {name: 'datatables',              files: [
              //'vendor/datatables/media/css/jquery.dataTables.css',
              'vendor/datatables/media/js/jquery.dataTables.min.js',
              //'vendor/datatables/media/css/dataTables.bootstrap.css',
              //'vendor/datatables/media/js/dataTables.bootstrap.js',
              'vendor/angular-datatables/dist/angular-datatables.min.js',
              'vendor/angular-datatables/dist/plugins/bootstrap/angular-datatables.bootstrap.min.js',
              'vendor/angular-datatables/dist/plugins/bootstrap/datatables.bootstrap.min.css'
            ], serie: true},
            {name: 'ngDeviceDetect',                 files: [
              'vendor/ng-device-detector/ng-device-detector.min.js',
              'vendor/re-tree/re-tree.min.js'
            ]},
            /*{name: 'ngClipboard',                 files: [
              'vendor/clipboard/dist/clipboard.js',
              'vendor/ngclipboard/dist/ngclipboard.js'
            ]},*/
          ]
        })
        ;

})();
