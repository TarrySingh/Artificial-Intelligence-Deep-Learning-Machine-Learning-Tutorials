use Module::Build;

my $build = Module::Build->new(
   module_name => 'ANTLR::Runtime',
   license     => 'perl',
   requires    => {
       'Carp'              => '1.04',
       'Readonly'          => '1.03',
       'List::MoreUtils'   => '0.22',
       'Want'              => '0.18',
       'Error'             => '0.17014',
   },
   build_requires => {
       'Test::Class'       => '0.28',
       'Java::JVM::Classfile' => '0.20',
   },
   test_files => 't/classes/ANTLR/Runtime/*.t t/*.t t/examples/*.t',
);
$build->create_build_script;
